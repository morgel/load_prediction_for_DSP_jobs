import argparse
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

import sys
import os
sys.path.append(os.path.join(os.path.abspath('')))
from helpers.models import MyGRU, MyCNN, MyCNNGRU
from helpers.trackers import AccuracyTracker1


parser = argparse.ArgumentParser()

parser.add_argument("-dp", "--dataset-path", type=str,
                    required=True, help="Path to dataset file.")
parser.add_argument("-dn", "--dataset-name", type=str,
                    required=True, help="Name of the dataset.")
parser.add_argument("-dtc", "--dataset-target-column", type=str,
                    required=True, help="Dataset target column.")
parser.add_argument("-dd", "--dataset-delimiter", type=str,
                    default="|", help="Dataset delimiter.")

parser.add_argument("-m", "--model", required=True, type=str, choices=["MyGRU", "MyCNN", "MyCNNGRU"], help="Model class to use.")

parser.add_argument("-d", "--device", type=str, required=True,
                    help="If available, will use this device for training.")
parser.add_argument("-v", "--verbose", action='store_true',
                    help="Enables printing of details during learning.")

parser.add_argument("-pcp", "--previous-checkpoint", action="append", type=lambda kv: kv.split("="), dest="checkpoint",
                    help="Previous checkpoint key-value pairs to resume training from. Must have pairs for 'path' and 'unique_log_name'.")
parser.add_argument("-cpi", "--checkpoint-interval", type=int, default=1000,
                    help="Every -cpi datapoints, a model checkpoint is stored.")

parser.add_argument("-w", "--window", type=int, default=3600,
                    help="Total amount of data to incorporate into training.")
parser.add_argument("-ts", "--test-split", type=float, default=0.2,
                    help="Fraction of window used for testing. Also maximal forecasting range.")

parser.add_argument("-in", "--intervals", type=str, required=True,
                    help="Comma-separated string of intervals. Will be parsed to list of integers.")

parser.add_argument("-e", "--epochs", type=int, nargs="+", default=[50, 10],
                    help="Number of epochs per update. The first value is for the initial training, the second for consecutive updates.")
parser.add_argument("-bs", "--batch_size", type=int, default=128,
                    help="Batch size during training / updates.")

parser.add_argument("-sl", "--sequence-length", type=int,
                    default=100, help="Length of each sequence inputted to the model.")
parser.add_argument("-hd", "--hidden-dimension", type=int,
                    default=16, help="Size of the hidden dimension.")

parser.add_argument("-nl", "--num-hidden-layers", type=int,
                    default=2, help="Number of hidden layers.")
parser.add_argument("-nck", "--num-conv-kernels", type=int,
                    default=32, help="Number of convolutional kernels.")
parser.add_argument("-cks", "--conv-kernel-size", type=int,
                    default=7, help="Kernel size of convolutional kernels.")
parser.add_argument("-pks", "--pool-kernel-size", type=int,
                    default=3, help="Kernel size of max pooling kernels.")


parser.add_argument("-do", "--dropout", type=float,
                    default=0.5, help="Dropout rate.")

parser.add_argument("-lr", "--learning-rate", type=float,
                    default=0.01, help="Learning rate.")
parser.add_argument("-wd", "--weight-decay", type=float,
                    default=1e-5, help="Weight decay.")
parser.add_argument("-b", "--betas", type=int, nargs="+",
                    default=[0.9, 0.999], help="Betas for Adam Optimizer.")

parser.add_argument("-gt", "--granularity-type", type=str,
                    default="seconds", help="Granularity of simulation. Valid values are [seconds, minutes].")

parser.add_argument("-ga", "--granularity-args", type=int, nargs="+",
                    default=[], help="Arguments of granularity of simulation.")


parser.add_argument("-an", "--apply-normalization", action='store_true',
                    help="Specify whether data should be normalized to [0,1]. Predictions will be inverse transformed.")

args = parser.parse_args()


target_device = args.device
### OUR TRAINING CONFIGURATION #######
verbose = args.verbose
batch_size = args.batch_size
epochs = tuple(args.epochs)
######################################

if args.conv_kernel_size % 2 == 0:
    raise ValueError("Convolution kernel size must be odd.")

if args.pool_kernel_size % 2 == 0:
    raise ValueError("MaxPooling kernel size must be odd.")

### OUR MODEL CONFIGURATION #######
model_args = {
    "input_dim": args.sequence_length,
    "hidden_dim": args.hidden_dimension,
    "dropout": args.dropout,
    "num_layers": args.num_hidden_layers,
    "num_conv_kernels": args.num_conv_kernels, 
    "conv_kernel_size": args.conv_kernel_size,
    "pool_kernel_size": args.pool_kernel_size,
    "bidirectional": False,
    "device": target_device
}
optimizer_args = {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "betas": tuple(args.betas)
}
###################################

granularity_type = args.granularity_type
granularity_args = args.granularity_args

print("Granularity Type:", granularity_type)
print("Granularity Args:", granularity_args)

dataset_path = args.dataset_path

base_df = pd.read_csv(
    dataset_path, delimiter=args.dataset_delimiter)


orig_values = base_df[args.dataset_target_column].values.reshape(-1, 1)
if granularity_type == "minutes":
    orig_values = orig_values[np.arange(len(orig_values)) % 60 == 0]
elif granularity_type == "seconds" and isinstance(granularity_args, list):
    orig_values = orig_values[granularity_args[0]:granularity_args[1]]
print(orig_values.shape)

previous_checkpoint = None
args_checkpoint = dict(args.checkpoint or {})
if isinstance(args_checkpoint.get("path", None), str) and isinstance(args_checkpoint.get("unique_log_name", None), str):
    previous_checkpoint = args_checkpoint
    

model_class = None
if args.model == "MyGRU":
    model_class = MyGRU
elif args.model == "MyCNN":
    model_class = MyCNN
elif args.model == "MyCNNGRU":
    model_class = MyCNNGRU         
    
ac = AccuracyTracker1({
    "dataset_path": dataset_path,
    "dataset_name": args.dataset_name,
    "data": orig_values,
    "model_class": model_class,
    "model_args": model_args,
    "optimizer_class": Adam,
    "optimizer_args": optimizer_args,
    "verbose": verbose,
    "epochs": epochs,
    "apply_normalization": args.apply_normalization,
    "batch_size": batch_size,
    "device": target_device,
    "window": args.window,
    "test_split": args.test_split,
    "intervals": [int(val.strip()) if "inf" not in val else np.inf for val in args.intervals.split(",")],
    "checkpoint_interval": args.checkpoint_interval,
    "previous_checkpoint": previous_checkpoint,
    "granularity_type": granularity_type,
    "granularity_args": granularity_args
})

ac.evaluate()
