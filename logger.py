import neptune
import os
import json
import torch
import numpy as np
import scipy.stats
import pandas as pd
import statistics
import matplotlib.pyplot as plt


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
            print(f"{task_name} {dict_name}: {str(input_dict)}")

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

    def clean_dict_for_json(self, input_dict):
        output_dict = {}
        for key, value, in input_dict.items():
            if isinstance(value, torch.Tensor):
                output_dict[key] = input_dict[key].cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                output_dict[key] = input_dict[key].tolist()
            elif isinstance(value, dict):
                output_dict[key] = self.clean_dict_for_json(input_dict[key])
            elif isinstance(value, list) or isinstance(value, str) or isinstance(value, float) or isinstance(value, int):
                output_dict[key] = value
            else:
                print(f"Not outputting {str(key)}: {str(value)}")
        return output_dict

    def save_avg_f1_graph(self, df, experiment_name):
        # Create 2 plots, one taking up most of the vertical axis, and a small table beneath that
        fig = plt.figure(figsize=(11,6), constrained_layout=True)
        gs = fig.add_gridspec(20, 1)
        ax0 = fig.add_subplot(gs[0:15, :])
        ax1 = fig.add_subplot(gs[16:, :])

        # Make an array which is of the same length as the number of datasets in the df
        x = np.array(list(range(len(df.index))))

        # How wide the bars should be
        width = 0.3

        # Plot a bar for both dnn and classical for each dataset, with the error for each dataset
        dnn_bar = ax0.bar(x-(width/2), df["DNN average F1"], width=width, align='center', yerr=df["DNN average F1 stdev"])
        classical_bar = ax0.bar(x+(width/2), df["Classical average F1"], width=width, align='center', yerr=df["Classical average F1 stdev"])
        ax0.legend([dnn_bar, classical_bar], ["DNN", "Classical"], loc=1)
        ax0.set_xticks(x)
        ax0.set_xticklabels(df.index)
        ax0.set_xlabel("Dataset")
        ax0.set_ylabel("Average F1")

        # For the table, we hide axes
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)

        # We get the statistical test data only
        stat_test_df = df[["T-test p val", "Wilcoxon p val"]]

        # We colour the cell green if the statistical test falls below our threshold
        certainty_threshold = 0.05
        passes_test_df = stat_test_df < certainty_threshold
        cell_colours = passes_test_df.applymap(lambda x: "#ace3ad" if x else "white")

        table = ax1.table(
                cellText=stat_test_df.applymap('{:,.3f}'.format).T.values.tolist(),
                rowLabels=["T-test p val", "Wilc p val"],
                cellColours=cell_colours.T.values.tolist(),
                loc='center',
                cellLoc='center'
                )

        table.scale(1, 1.5)

        ax1.axis('off')

        # Save file as png
        file_name = os.path.join(self.output_dir_name, f"{experiment_name}.png")
        fig.savefig(file_name)
        return file_name

    def log_image(self, image_name, image_path):
        neptune.log_image(image_name, image_path)

    def log_json(self, file_name, input_dict):
        # Log supplied dict to a json file

        # Make sure that dict does not contain GPU Tensors
        input_dict = self.clean_dict_for_json(input_dict)

        if not os.path.exists(self.output_dir_name):
            os.makedirs(self.output_dir_name)

        with open(os.path.join(self.output_dir_name, file_name),"w") as f:
            json.dump(input_dict, f)

    def log_output_files(self, experiment_name, experiment_number):
        # Create a new meta-experiment in which to output the results of all runs of experiment
        neptune.create_experiment(name=experiment_name)

        all_results_dict = {}

        # Cycle through all files in output folder, and look for only .json output files
        for filename in os.listdir(self.output_dir_name):
            if filename.endswith(".json"):
                # Open json files and read into dict
                 with open(os.path.join(self.output_dir_name, filename), "r") as f:
                     json_data_dict = json.load(f)

                     all_results_dict[filename[:-5]] = json_data_dict

                     # Log the read file under the filename (minus suffix)
                     self.log_dict(filename[:-5], json_data_dict)

        self.output_topline_results(all_results_dict, experiment_number, experiment_name)

        self.stop()

    def output_topline_results(self, results_dict, experiment_number, experiment_name):
        if experiment_number == 1:
            self.log_experiment_1(results_dict, experiment_name)
        elif experiment_number == 2:
            self.log_experiment_2(results_dict, experiment_name)
        elif experiment_number == 3:
            self.log_experiment_3(results_dict, experiment_name)
        elif experiment_number == 4:
            self.log_experiment_4(results_dict, experiment_name)
        else:
            raise Exception(f"Experiment number {experiment_number} not supported")

    def log_experiment_1(self, results_dict, experiment_name):
        # Logs the average average f1 for each dataset over the x number of runs from different random seeds

        classical_results_names = [run_name for run_name in results_dict.keys() if "best_classical_baselines" in run_name]
        ft_results_names = [run_name for run_name in results_dict.keys() if "ft_task_test_metrics" in run_name]

        # Gets the dataset names (E.g. di_sorbo_2017, maalej_2016) from the files in the output folder
        dataset_names = list(set(["_".join(run_name.replace("_ft_task_test_metrics","").split("_")[:-1]) for run_name in ft_results_names]))

        pan_dataset_results_list = []

        for dataset in dataset_names:
            classical_run_values = []
            dnn_run_values = []

            classical_dataset_runs = [run_name for run_name in classical_results_names if dataset in run_name]
            dnn_dataset_runs = [run_name for run_name in ft_results_names if dataset in run_name]

            for i, (dnn_dataset_run, classical_dataset_run) in enumerate(zip(dnn_dataset_runs, classical_dataset_runs)):
                classical_target_value = results_dict[classical_dataset_run]["multiclass"]["all best metrics"]["test results best"]["average f1"]
                dnn_target_value = results_dict[dnn_dataset_run][dataset]["average f1"]

                self.log_metric(f"classical {dataset}", i, classical_target_value)
                self.log_metric(f"dnn {dataset}", i, dnn_target_value)

                classical_run_values.append(classical_target_value)
                dnn_run_values.append(dnn_target_value)

            # Take the list of the average f1 scores for each run of the dataset with a different random split
            # and get the average of these values, for both classical and DNN.
            # Also calculate the standard deviation of these values for a view as to the stability of these results.
            averaged_classical_target_value = statistics.mean(classical_run_values)
            averaged_dnn_target_value = statistics.mean(dnn_run_values)
            averaged_classical_target_value_sd = statistics.stdev(classical_run_values)
            averaged_dnn_target_value_sd = statistics.stdev(dnn_run_values)

            ttest_p_val = scipy.stats.ttest_rel(dnn_run_values, classical_run_values).pvalue
            wilcoxon_p_val = scipy.stats.wilcoxon(dnn_run_values, classical_run_values, alternative="two-sided").pvalue

            pan_dataset_results_list.append(
                                        {"Classical average F1": averaged_classical_target_value,
                                        "DNN average F1": averaged_dnn_target_value,
                                        "Classical average F1 stdev": averaged_classical_target_value_sd,
                                        "DNN average F1 stdev": averaged_dnn_target_value_sd,
                                        "T-test p val": ttest_p_val,
                                        "Wilcoxon p val": wilcoxon_p_val})

            # Add wilcoxon for each population https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.wilcoxon.html

            # Output bar graph of all datasets with SD error bars

            neptune.log_text("Overall results", f"{dataset} DNN: {averaged_dnn_target_value}")
            neptune.log_text("Overall results", f"{dataset} Classical: {averaged_classical_target_value}")
            neptune.log_text("Overall results", f"{dataset} DNN stdev: {averaged_dnn_target_value_sd}")
            neptune.log_text("Overall results", f"{dataset} Classical stdev: {averaged_classical_target_value_sd}")
            neptune.log_text("Overall results", f"{dataset} t-test p val: {ttest_p_val}")
            neptune.log_text("Overall results", f"{dataset} Wilcoxon p val: {wilcoxon_p_val}")
            neptune.log_text("Overall results", "\n")

        results_df = pd.DataFrame(pan_dataset_results_list, index=dataset_names)

        graph_path = self.save_avg_f1_graph(results_df, experiment_name)
        self.log_image(f"{experiment_name} graphical results", graph_path)


    def log_experiment_2(self, results_dict, experiment_name):
        raise Exception(f"Experiment 2 output not yet implemented")

    def log_experiment_3(self, results_dict, experiment_name):
        raise Exception(f"Experiment 3 output not yet implemented")

    def log_experiment_4(self, results_dict, experiment_name):
        raise Exception(f"Experiment 4 output not yet implemented")
