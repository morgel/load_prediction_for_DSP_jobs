import torch
import zipfile
import os
import datetime
import numpy as np
import pandas as pd
import re
import copy
import json

from helpers.printer import Printer


class BaseCheckpointer(Printer):
    def __init__(self, configuration={}):
        super(BaseCheckpointer, self).__init__()
        self.verbose = False

        self.configuration = configuration
        self.device = configuration.get("device")
        self.dataset_name = configuration.get("dataset_name")

        self.granularity_type = configuration.get(
            "granularity_type", "seconds")

        self.model_class = configuration.get("model_class")
        self.model_args = configuration.get("model_args", {})

        self.optimizer_class = configuration.get("optimizer_class")
        self.optimizer_args = configuration.get("optimizer_args", {})

        ### ensure each specialized model starts with same weight initialization ###
        current_meta = self._init_guardians()

        loc_model_class = str(current_meta["current_model"])
        loc_model_args = json.dumps(
            current_meta["current_model"].get_parameter_dict())

        for key in list(current_meta.keys()):
            if "model" in key or "optimizer" in key:
                current_meta[key] = current_meta[key].state_dict()

        self.checkpoints = {
            "root": current_meta
        }
        ############################################################################

        previous_checkpoint = configuration.get("previous_checkpoint", None)
        if isinstance(previous_checkpoint, dict):
            self.unique_log_name = previous_checkpoint["unique_log_name"]
            self._print_line("#" * 10, "Loading models / optimizers...")
            self.load_models(previous_checkpoint)
            self.previous_checkpoint = True
        else:
            self.unique_log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.previous_checkpoint = False
            self._print_csv(*self.csv_headers)

        self._print_line("Model class:", loc_model_class)
        self._print_line("Model args:", loc_model_args)

    @property
    def csv_headers(self):
        raise NotImplementedError("Subclasses should implement this!")

    def create_model_key(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def create_checkpoint_dict(self, current_meta, next_meta, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def filter_condition(self, df: pd.DataFrame, dictt: dict):
        raise NotImplementedError("Subclasses should implement this!")

    def delete_condition(self, df: pd.DataFrame, dictt: dict):
        raise NotImplementedError("Subclasses should implement this!")

    def _init_guardians(self):
        current_model = self.model_class(
            **self.model_args).double().to(self.device)
        current_optimizer = self.optimizer_class(
            current_model.parameters(), **self.optimizer_args)

        next_model = self.model_class(
            **self.model_args).double().to(self.device)
        next_optimizer = self.optimizer_class(
            next_model.parameters(), **self.optimizer_args)

        current_meta = dict.fromkeys(["last_update_idx", "available_at_idx",
                                      "next_update_idx", "next_model", "next_optimizer", "interval", "next_version"])

        current_meta.update(current_model=current_model, current_optimizer=current_optimizer,
                            next_model=next_model, next_optimizer=next_optimizer, current_version=0)

        return current_meta

    def get_guardians(self, *args, **kwargs):
        current_meta = self._init_guardians()

        key = self.create_model_key(**kwargs)
        inner_dict = copy.deepcopy(
            self.checkpoints[key if key in self.checkpoints else "root"])

        for k in list(inner_dict.keys()):
            if "model" in k or "optimizer" in k:
                if inner_dict[k] is not None:
                    current_meta[k].load_state_dict(inner_dict[k])
                else:
                    current_meta[k] = inner_dict[k]  # assign None
            else:
                current_meta[k] = inner_dict[k]

        return copy.deepcopy(current_meta)

    def set_guardians(self, next_meta, **kwargs):
        key = self.create_model_key(**kwargs)
        current_meta = copy.deepcopy(
            self.checkpoints[key if key in self.checkpoints else "root"])
        self.checkpoints[key] = self.create_checkpoint_dict(
            current_meta, next_meta, **kwargs)

    def load_models(self, previous_checkpoint):
        self._print_line(
            "#" * 10, "Resuming from Checkpoint. Loading previous models...")
        unique_log_name = previous_checkpoint["unique_log_name"]
        path = previous_checkpoint["path"]

        extraction_regex = re.compile(f'{unique_log_name}_(.*)\\.tar')

        for root, _, filenames in os.walk(path):
            for name in filenames:
                matching = extraction_regex.match(name)
                if matching is not None:
                    key = matching.group(1)

                    name_path = os.path.join(root, name)
                    name_path = os.path.normpath(name_path)

                    self.checkpoints[key] = torch.load(name_path)

        # delete all entries written since last successful checkpoint
        for key, value in self.checkpoints.items():
            df = pd.read_csv(f"{unique_log_name}_results.csv", delimiter="|")

            sub_df = df.loc[self.filter_condition(df, value), :]
            if value.get("last_update_idx", None) is not None:
                sub_df = sub_df.loc[self.delete_condition(sub_df, value), :]
                df.drop(sub_df.index, inplace=True)
                df.to_csv(f"{unique_log_name}_results.csv",
                          sep="|", index=False)

    def save_models(self, index):
        self._print_line(
            "#" * 10, f"Saving models / optimizers... [Sample Index: {index}]")
        for key, value in self.checkpoints.items():
            torch.save(value, "{}_{}.tar".format(self.unique_log_name, key))

    def zip_files(self):
        self._print_line("#" * 10, "Zipping files...")

        loc_model_class = str(self.get_guardians(
            interval="root")["next_model"])
        loc_device = "CPU" if "cpu" == self.device else "GPU"

        with zipfile.ZipFile(f"{self.dataset_name}_{self.unique_log_name}_{loc_model_class}_{loc_device}_{self.granularity_type}.zip",
                             "w",
                             zipfile.ZIP_DEFLATED,
                             allowZip64=True) as zf:
            for root, _, filenames in os.walk("."):
                for name in filenames:
                    if self.unique_log_name in name and ".zip" not in name:
                        name = os.path.join(root, name)
                        name = os.path.normpath(name)
                        zf.write(name, name)
                        os.remove(name)

###############################################################################
###############################################################################
###############################################################################


class StateCheckpointer1(BaseCheckpointer):
    def __init__(self, configuration={}):
        super(StateCheckpointer1, self).__init__(configuration=configuration)

    @property
    def csv_headers(self):
        return ["interval", "f_length", "version", "p_number", "s_number",
                "mse", "smape", "gpu", "util", "duration"]

    def filter_condition(self, df: pd.DataFrame, dictt: dict):
        return df["interval"] == dictt.get("interval")

    def delete_condition(self, df: pd.DataFrame, dictt: dict):
        return df["s_number"] > dictt.get("last_update_idx")

    def create_model_key(self, *args, **kwargs):
        value = kwargs.get('interval')
        if value == "root":
            return value
        else:
            return f"interval={value}"

    def create_checkpoint_dict(self, current_meta, next_meta, **kwargs):

        next_meta["interval"] = kwargs.get("interval")

        for key in list(next_meta.keys()):
            if "model" in key or "optimizer" in key:
                if next_meta[key] is not None:
                    next_meta[key] = next_meta[key].state_dict()

        current_meta.update(next_meta)

        return current_meta
