import numpy as np
from sklearn.preprocessing import MinMaxScaler
from helpers.preparations import create_dataset, get_data_loaders
from helpers.losses import SymmetricMeanAbsolutePercentageErrorLoss
from helpers.utils import GPUMonitor
from helpers.printer import Printer
from helpers.checkpointers import StateCheckpointer1
import torch
import torch.nn as nn
import datetime
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from ignite.handlers import TerminateOnNan
from helpers.losses import SymmetricMeanAbsolutePercentageError
import psutil
import math
import pandas as pd
import zipfile
import os
import copy
import re
import functools


def measure(func):
    """ Measure duration of utilization of decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        duration = None
        util = None

        is_gpu, device, granularity_type = kwargs.get(
            "is_gpu", False), kwargs.get("device", "cpu"), kwargs.get("granularity_type", "seconds")

        if is_gpu:
            # Instantiate monitor with a 0.03-second delay between updates
            monitor = GPUMonitor(
                0.05, int(device.split(":")[1]))
        else:
            p = psutil.Process()
            p.cpu_percent(interval=None)

        start = datetime.datetime.now()
        #################################
        response = func(*args, **kwargs)
        #################################
        end = datetime.datetime.now()

        duration = (end - start).total_seconds()
        if granularity_type == "minutes":
            duration /= 60

        if is_gpu:
            monitor.stop()
            util = monitor.gpu_util
        else:
            util = p.cpu_percent(
                interval=None) / psutil.cpu_count()
            util /= 100 # to interval [0,1]

        return duration, util, response

    return wrapper


class BaseTracker(Printer):
    def __init__(self, configuration={}):
        super(BaseTracker, self).__init__()

        self.checkpointer = self._init_checkpointer(configuration)

        self.unique_log_name = self.checkpointer.unique_log_name

        self.verbose = configuration.get("verbose", False)
        self._print_line("Verbose: ", self.verbose)

        self.checkpoint_interval = configuration.get(
            "checkpoint_interval", 1000)
        self._print_line("Checkpoint interval: ", self.checkpoint_interval)

        self._print_line("Dataset name: ", configuration.get(
            "dataset_name"))
        
        self._print_line("Dataset path: ", configuration.get(
            "dataset_path"))

        self.granularity_type = configuration.get(
            "granularity_type", "seconds")
        self._print_line("Granularity type:", self.granularity_type)

        self.granularity_args = configuration.get("granularity_args", None)
        self._print_line("Granularity args:", self.granularity_args)

        self.window = configuration.get("window", 3600)
        self._print_line("Total window: ", self.window)

        self.test_split = configuration.get("test_split", 0.2)
        self._print_line("Test split: ", self.test_split)

        self.data = configuration.get("data", np.array([]))
        self._print_line("#Data points: ", len(self.data))

        self.epochs = configuration.get("epochs", (20, 10))
        self._print_line("#Epochs: ", self.epochs)

        self.batch_size = configuration.get("batch_size", 128)
        self._print_line("Batch size: ", self.batch_size)

        self.criterion = configuration.get(
            "criterion", SymmetricMeanAbsolutePercentageErrorLoss())
        self._print_line("Criterion: ", self.criterion)

        self.device = configuration.get("device")
        self._print_line("Device: ", self.device)
        self.is_gpu = "cuda" in self.device

        self.smape_loss = SymmetricMeanAbsolutePercentageErrorLoss()
        self.mse_loss = nn.MSELoss()

        self.seq_length = configuration.get(
            "model_args", {}).get("input_dim", 100)

        self.intervals = configuration.get(
            "intervals", [0, 5, 15, 30, 60, np.inf])
        self._print_line("Intervals: ", self.intervals)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.apply_normalization = configuration.get("apply_normalization")
        self._print_line("Apply Normalization: ", self.apply_normalization)

        self._print_line("Previous Checkpoint: ",
                         self.checkpointer.previous_checkpoint)

    def _prepare_data(self, train_indices, test_indices):
        train = self.data[train_indices]
        test = self.data[test_indices]

        train_scaled = self.scaler.fit_transform(train)
        test_scaled = self.scaler.transform(test)
        
        if not self.apply_normalization:
            train_scaled = self.scaler.inverse_transform(train_scaled)
            test_scaled = self.scaler.inverse_transform(test_scaled)

        trainX, trainY = create_dataset(
            train_scaled, self.seq_length, device=self.device)

        train_loader = get_data_loaders(
            trainX,
            trainY,
            batch_size=self.batch_size
        )

        return train_loader, train_scaled, test

    def _init_checkpointer(self, configuration):
        raise NotImplementedError("Subclasses should implement this!")

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement this!")

    def _train(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def _predict(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

#################################################################################
#################################################################################
#################################################################################


class AccuracyTracker1(BaseTracker):
    def __init__(self, configuration={}):
        super(AccuracyTracker1, self).__init__(configuration=configuration)

    def _init_checkpointer(self, configuration):
        return StateCheckpointer1(configuration=configuration)

    @measure
    def _train(self, model, optimizer, train_loader, max_epochs, **kwargs):
        trainer = create_supervised_trainer(
            model, optimizer, self.criterion)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan())

        val_metrics = {
            "smape": SymmetricMeanAbsolutePercentageError()
        }
        evaluator = create_supervised_evaluator(model, metrics=val_metrics)

        @trainer.on(Events.COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            self._print_line("Interval {} Training Results - Epoch: {} Avg smape: {:.2f}"
                             .format(kwargs.get("interval"), trainer.state.epoch, metrics["smape"]))

        trainer.run(train_loader, max_epochs=max_epochs)
        return (model, optimizer)

    @measure
    def _predict(self, model, test_size, starting_point, **kwargs):
        data_collect = []
        model.eval()
        with torch.no_grad():
            for _ in range(test_size):
                value = model.forward_eval_single(starting_point)
                data_collect.append(value.cpu().detach().item())
                starting_point = torch.cat(
                    (starting_point[:, 1:], value.view(1, -1)), dim=1)
        model.h_previous = None
        return data_collect, model

    def evaluate(self):
        self._print_line("#" * 10, "Start evaluation...")

        data_length = len(self.data)
        train_size = self.window
        self._print_line("Train size:", train_size)

        test_size = int(int(self.window / (1 - self.test_split))
                        * self.test_split)
        self._print_line("Test size:", test_size)

        for sample_idx in range(train_size, data_length-test_size):

            train_loader, train_scaled, test = self._prepare_data(
                np.arange((sample_idx - train_size),
                          sample_idx),  # train indices
                # test indices
                np.arange(sample_idx, (sample_idx + test_size))
            )

            #############################
            #############################
            # INITIAL MODEL TRAINING
            #############################
            #############################
            if sample_idx == self.window:
                self._print_line("Initial model training...")
                current_meta = self.checkpointer.get_guardians(
                    interval="root")

                current_model = current_meta.get("current_model")
                current_optimizer = current_meta.get("current_optimizer")

                # only the case if resumed training. skip already inspected data points
                if current_meta.get("last_update_idx", None) is not None and sample_idx <= current_meta.get("last_update_idx"):
                    continue

                train_duration, train_util, (current_model, current_optimizer) = self._train(
                    current_model,
                    current_optimizer,
                    train_loader,
                    self.epochs[0],
                    is_gpu=self.is_gpu,
                    device=self.device,
                    granularity_type=self.granularity_type,
                    interval="root"
                )

                current_meta.update({
                    "current_version": 0,
                    "current_model": None,
                    "current_optimizer": None,
                    "next_version": 1,
                    "next_model": current_model,
                    "next_optimizer": current_optimizer,

                })

                self.checkpointer.set_guardians(current_meta, interval="root")

                for interval in self.intervals:
                    self._print_csv(
                        interval,
                        -1,
                        0,
                        sample_idx,
                        sample_idx,
                        -1,
                        -1,
                        int(self.is_gpu),
                        "{:.3f}".format(train_util),
                        "{:.3f}".format(train_duration)
                    )

                    interval_meta = self.checkpointer.get_guardians(
                        interval=interval)

                    interval_meta.update({
                        "last_update_idx": sample_idx,
                        "next_update_idx": sample_idx + (max(interval, math.ceil(train_duration) if np.isfinite(interval) else np.inf)),
                        "available_at_idx": sample_idx + (min(max(interval, math.ceil(train_duration) if np.isfinite(interval) else 1), math.ceil(train_duration) if np.isfinite(interval) else 1))
                    })

                    self.checkpointer.set_guardians(
                        interval_meta, interval=interval)

                self._print_line("Initial model training completed.")
            #############################
            #############################
            # CHECK INTERVALS
            #############################
            #############################
            else:
                for interval in self.intervals:

                    current_meta = self.checkpointer.get_guardians(
                        interval=interval)

                    current_model = current_meta.get("current_model")
                    current_optimizer = current_meta.get("current_optimizer")

                    next_model = current_meta.get("next_model")
                    next_optimizer = current_meta.get("next_optimizer")

                    # only the case if resumed training. skip already inspected data points
                    if current_meta.get("last_update_idx", None) is not None and sample_idx <= current_meta.get("last_update_idx"):
                        continue

                    #############################
                    # swap models
                    #############################
                    if current_meta.get("available_at_idx", None) is not None and sample_idx == current_meta.get("available_at_idx"):
                        current_model = next_model
                        current_optimizer = next_optimizer

                        current_meta.update(
                            current_version=current_meta.get(
                                "current_version") + 1,
                            next_version=current_meta.get("next_version") + 1,
                            current_model=current_model,
                            current_optimizer=current_optimizer)

                    #############################
                    # model did not yet finish initial training
                    #############################
                    if current_model is None and current_optimizer is None:
                        continue

                    current_meta_copy = copy.deepcopy(current_meta)

                    #############################
                    # prediction / forecasting
                    #############################
                    starting_point = torch.from_numpy(
                        train_scaled[-self.seq_length:].reshape(1, -1)).to(self.device)

                    pred_duration, pred_util, (data_collect, current_model) = self._predict(
                        current_model,
                        test_size,
                        starting_point,
                        is_gpu=self.is_gpu,
                        device=self.device,
                        granularity_type=self.granularity_type
                    )

                    test_pred = np.array(data_collect).reshape(-1, 1)
                    
                    if self.apply_normalization:
                        test_pred = self.scaler.inverse_transform(test_pred)

                    for f in range(int(test_size / 10), test_size + 1, int(test_size / 10)):
                        local_pred = torch.from_numpy(test_pred[:f])
                        local_true = torch.from_numpy(test[:f])
                        smape_score = self.smape_loss(
                            local_pred, local_true).item()
                        mse_score = self.mse_loss(
                            local_pred, local_true).item()

                        self._print_csv(
                            interval,
                            f,
                            current_meta_copy.get("current_version"),
                            current_meta.get("last_update_idx"),
                            sample_idx,
                            "{:.3f}".format(mse_score),
                            "{:.3f}".format(smape_score),
                            int(self.is_gpu),
                            "{:.3f}".format(pred_util),
                            "{:.3f}".format(pred_duration)
                        )
                    #############################
                    # (re-)training
                    #############################
                    if current_meta.get("next_update_idx", None) is not None and sample_idx == current_meta.get("next_update_idx"):

                        train_duration, train_util, (current_model, current_optimizer) = self._train(
                            current_model,
                            current_optimizer,
                            train_loader,
                            self.epochs[1],
                            is_gpu=self.is_gpu,
                            device=self.device,
                            granularity_type=self.granularity_type,
                            interval=interval
                        )

                        self._print_csv(
                            interval,
                            -1,
                            current_meta_copy.get("current_version"),
                            sample_idx,
                            sample_idx,
                            -1,
                            -1,
                            int(self.is_gpu),
                            "{:.3f}".format(train_util),
                            "{:.3f}".format(train_duration)
                        )

                        current_meta.update({
                            "current_model": current_meta_copy.get("current_model"),
                            "current_optimizer": current_meta_copy.get("current_optimizer"),
                            "last_update_idx": sample_idx,
                            "next_update_idx": sample_idx + (max(interval, math.ceil(train_duration) if np.isfinite(interval) else np.inf)),
                            "available_at_idx": sample_idx + (min(max(interval, math.ceil(train_duration) if np.isfinite(interval) else 1), math.ceil(train_duration) if np.isfinite(interval) else 1)),
                            "next_model": current_model,
                            "next_optimizer": current_optimizer
                        })

                        self.checkpointer.set_guardians(
                            current_meta, interval=interval)

                    else:
                        self.checkpointer.set_guardians(
                            current_meta_copy, interval=interval)

            if sample_idx % self.checkpoint_interval == 0:
                self.checkpointer.save_models(sample_idx)

        self.checkpointer.save_models(data_length - test_size)
        self._print_line("#" * 10, "Evaluation completed.")

        self.checkpointer.zip_files()
        print("#" * 10, "Done.", "#" * 10)
