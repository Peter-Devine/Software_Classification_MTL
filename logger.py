import neptune
import os
import json
import torch

class NeptuneLogger:
    def __init__(self, username):
        self.output_dir_name = "output"

        if len(username)>0:
            # HAVE YOUR API KEY SAVED AS AN ENV VAR $NEPTUNE_API_TOKEN (or you can provide it as a "api_token" argument below if your code is private)
            neptune.init(username + '/sandbox')
            self.logger_active = True
        else:
            self.logger_active = False

    def create_experiment(self, PARAMS):
        if self.logger_active:
            neptune.create_experiment(name="__|__".join(PARAMS.dataset_name_list),
                              params=vars(PARAMS))
        else:
            dataset_list_str = ", ".join(PARAMS.dataset_name_list)
            print(f"Now outputting experimental data for the experiment with {dataset_list_str} datasets")

    def log_metric(self, metric_name, x, y):
        if self.logger_active:
            neptune.log_metric(metric_name, x, y)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \nmetric:{str(y)}\n\n")

    def is_numeric(self, value):
        try:
            float(value)
            return True
        except Exception:
            return False

    def log_array(self, metric_name, x, array):
        if self.logger_active:
            for i, cell in enumerate(array):
                inner_metric_name = f"{metric_name}__{str(i)}"
                if self.is_numeric(cell):
                    neptune.log_metric(inner_metric_name, x, cell)
                else:
                    self.log_array(inner_metric_name, x, cell)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \narray:{str(array)}\n\n")

    def log_text(self, metric_name, x, text):
        if self.logger_active:
            neptune.log_text(metric_name, x, text)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \ntext:{text}\n\n")

    def log_dict(self, dict_name, input_dict, task_name="", recursion_level = 0):
        if self.logger_active:
            # Add spaces so that dict prints prettily in logger
            spacing_str = '|' + ' - - - '*recursion_level
            for key, value in input_dict.items():
                if type(value) == dict:
                    neptune.log_text(f"{task_name} {dict_name}", f"{spacing_str}{str(key)}")
                    self.log_dict(dict_name, value, task_name, recursion_level+1)
                else:
                    neptune.log_text(f"{task_name} {dict_name}", f"{spacing_str}{str(key)}: {str(value)}")
        else:
            print(f"{task_name} {dict_name}: {str(dict)}")

    def log_results(self, task_name, split_type, epoch, results_dict):
        if self.logger_active:
            metric_prefix = f"{task_name} {split_type} "
            for metric_name, metric in results_dict.items():
                if self.is_numeric(metric):
                    self.log_metric(metric_prefix + metric_name, epoch, metric)
                else:
                    self.log_text(metric_prefix + metric_name, epoch, str(metric))

    def stop(self):
        if self.logger_active:
            neptune.stop()

    def clean_dict_for_json(self, dict):
        for key, value, in dict.items():
            if isinstance(value, torch.Tensor):
                dict[key] = dict[key].cpu().numpy()
            elif isinstance(value, dict):
                dict[key] = self.clean_dict_for_json(dict[key])
        return dict

    def log_json(self, file_name, dict):
        # Log supplied dict to a json file

        # Make sure that dict does not contain GPU Tensors
        dict = self.clean_dict_for_json(dict)

        if not os.path.exists(self.output_dir_name):
            os.makedirs(self.output_dir_name)

        with open(os.path.join(self.output_dir_name, file_name),"w") as f:
            json.dump(dict, f)

    def log_output_files(self, experiment_name):
        # Create a new meta-experiment in which to output the results of all runs of experiment
        neptune.create_experiment(name=experiment_name)

        # Cycle through all files in output folder, and look for only .json output files
        for filename in os.listdir(self.output_dir_name):
            if filename.endswith(".json"):
                # Open json files and read into dict
                 with open(os.path.join(self.output_dir_name, filename), "r") as f:
                     json_data_dict = json.load(f)

                     # Log the read file under the filename (minus suffix)
                     self.log_dict(filename[:-5], json_data_dict)

        self.stop()
