import scipy.stats
import pandas as pd
import statistics

def get_stats_for_two_lists(list_of_vals_0, list_of_vals_1):

    mean_0 = statistics.mean(list_of_vals_0) if len(list_of_vals_0) > 0 else 0
    stdev_0 = statistics.stdev(list_of_vals_0) if len(list_of_vals_0) > 0 else 0

    mean_1 = statistics.mean(list_of_vals_1) if len(list_of_vals_1) > 0 else 0
    stdev_1 = statistics.stdev(list_of_vals_1) if len(list_of_vals_1) > 0 else 0

    ttest_p_val = scipy.stats.ttest_rel(list_of_vals_0, list_of_vals_1).pvalue if len(list_of_vals_0) > 0 and len(list_of_vals_1) > 0 else 1
    wilcoxon_p_val = scipy.stats.wilcoxon(list_of_vals_0, list_of_vals_1, alternative="two-sided").pvalue if len(list_of_vals_0) > 0 and len(list_of_vals_1) > 0 else 1

    return mean_0, stdev_0, mean_1, stdev_1, ttest_p_val, wilcoxon_p_val

def get_indomain_single_task_results(results_dict, logger):
    # Logs the average average f1 for each dataset over the x number of runs from different random seeds
    classical_results_names = [run_name for run_name in results_dict.keys() if "single_task_best_classical_baselines" in run_name]
    ft_results_names = [run_name for run_name in results_dict.keys() if "single_task_ft_task_test_metrics" in run_name]

    # Gets the dataset names (E.g. di_sorbo_2017, maalej_2016) from the files in the output folder
    dataset_names = sorted(list(set(["_".join(run_name.replace("_single_task_ft_task_test_metrics","").split("_")[:-1]) for run_name in ft_results_names])))

    pan_dataset_results_list = []

    for counter, dataset in enumerate(dataset_names):
        classical_run_values = []
        dnn_run_values = []

        classical_dataset_runs =  sorted([run_name for run_name in classical_results_names if dataset in run_name])
        dnn_dataset_runs =  sorted([run_name for run_name in ft_results_names if dataset in run_name])

        for i, (dnn_dataset_run, classical_dataset_run) in enumerate(zip(dnn_dataset_runs, classical_dataset_runs)):
            classical_target_value = results_dict[classical_dataset_run]["multiclass"]["all best metrics"]["test results best"]["average f1"]
            dnn_target_value = results_dict[dnn_dataset_run][dataset]["average f1"]

            logger.log_metric(f"classical {dataset}", i, classical_target_value)
            logger.log_metric(f"dnn {dataset}", i, dnn_target_value)

            classical_run_values.append(classical_target_value)
            dnn_run_values.append(dnn_target_value)

        # Take the list of the average f1 scores for each run of the dataset with a different random split
        # and get the average of these values, for both classical and DNN.
        # Also calculate the standard deviation of these values for a view as to the stability of these results.
        # Do statistical tests to determine whether the two populations are statistically different
        # Add wilcoxon for each population https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.wilcoxon.html
        # Add 2 tailed paired t-test for each set of results
        stats = get_stats_for_two_lists(classical_run_values, dnn_run_values)
        averaged_classical_target_value, averaged_classical_target_value_sd, averaged_dnn_target_value, averaged_dnn_target_value_sd, ttest_p_val, wilcoxon_p_val = stats

        pan_dataset_results_list.append(
                                    {"Classical average F1": averaged_classical_target_value,
                                    "DNN average F1": averaged_dnn_target_value,
                                    "Classical average F1 stdev": averaged_classical_target_value_sd,
                                    "DNN average F1 stdev": averaged_dnn_target_value_sd,
                                    "In-domain T-test p val": ttest_p_val,
                                    "In-domain Wilcoxon p val": wilcoxon_p_val})


        logger.log_text("Overall results", counter, f"{dataset} DNN: {averaged_dnn_target_value}")
        logger.log_text("Overall results", counter, f"{dataset} Classical: {averaged_classical_target_value}")
        logger.log_text("Overall results", counter, f"{dataset} DNN stdev: {averaged_dnn_target_value_sd}")
        logger.log_text("Overall results", counter, f"{dataset} Classical stdev: {averaged_classical_target_value_sd}")
        logger.log_text("Overall results", counter, f"{dataset} t-test p val: {ttest_p_val}")
        logger.log_text("Overall results", counter, f"{dataset} Wilcoxon p val: {wilcoxon_p_val}")
        logger.log_text("Overall results", counter, "\n")

    results_df = pd.DataFrame(pan_dataset_results_list, index=dataset_names)

    return results_df

def get_outdomain_single_task_results(results_dict, logger):
    # Logs the average average f1 for each dataset over the x number of runs from different random seeds
    classical_results_names = [run_name for run_name in results_dict.keys() if "single_task_zero_shot_classical_baselines" in run_name]
    classical_binary_results_names = [run_name for run_name in results_dict.keys() if "single_task_mtl_zero_shot_classical_baselines" in run_name]
    ft_results_names = [run_name for run_name in results_dict.keys() if "single_task_zero_shot_test_metrics" in run_name]

    # Gets the dataset names (E.g. di_sorbo_2017, maalej_2016) from the files in the output folder
    dataset_names = sorted(list(set(["_".join(run_name.replace("_single_task_zero_shot_test_metrics","").split("_")[:-1]) for run_name in ft_results_names])))

    pan_dataset_results_list = []

    # Make a dictionary that holds a list of values for each test dataset
    classical_run_values = {}
    classical_binary_run_values = {}
    dnn_run_values = {}

    for i, dataset in enumerate(dataset_names):

        classical_dataset_runs = sorted([run_name for run_name in classical_results_names if dataset in run_name])
        classical_binary_dataset_runs = sorted([run_name for run_name in classical_binary_results_names if dataset in run_name])
        dnn_dataset_runs = sorted([run_name for run_name in ft_results_names if dataset in run_name])

        for dnn_dataset_run, classical_dataset_run, classical_bin_dataset_run in zip(dnn_dataset_runs, classical_dataset_runs, classical_binary_dataset_runs):

            ##### CLASSICAL #######

            # First, we get the dict of classical results for the model that has been trained on dataset
            classical_zero_shot_results = results_dict[classical_dataset_run][dataset]

            for test_dataset_name, test_dataset_results in classical_zero_shot_results.items():
                # Initialize the test dataset results list if it hasn't been initialized
                if test_dataset_name not in classical_run_values.keys():
                    classical_run_values[test_dataset_name] = {}
                    for train_dataset in dataset_names:
                        classical_run_values[test_dataset_name][train_dataset] = []

                target_value = test_dataset_results["average f1"]

                logger.log_metric(f"classical zero shot {test_dataset_name}", i, target_value)
                classical_run_values[test_dataset_name][dataset].append(target_value)


            ##### CLASSICAL BINARY ######

            classical_bin_zero_shot_results = results_dict[classical_bin_dataset_run]

            for test_dataset_name, test_dataset_results in classical_bin_zero_shot_results.items():
                # Initialize the test dataset results list if it hasn't been initialized
                if test_dataset_name not in classical_binary_run_values.keys():
                    classical_binary_run_values[test_dataset_name] = {}
                    for train_dataset in dataset_names:
                        classical_binary_run_values[test_dataset_name][train_dataset] = []

                target_value = test_dataset_results["average f1"]

                logger.log_metric(f"classical binary zero shot {test_dataset_name}", i, target_value)
                classical_binary_run_values[test_dataset_name][dataset].append(target_value)

            ##### DNN #####

            # We get the dict of DNN results for the model that has been trained on dataset
            dnn_zero_shot_results = results_dict[dnn_dataset_run][dataset]

            for test_dataset_name, test_dataset_results in dnn_zero_shot_results.items():
                # Initialize the test dataset results list if it hasn't been initialized
                if test_dataset_name not in dnn_run_values.keys():
                    dnn_run_values[test_dataset_name] = {}
                    for train_dataset in dataset_names:
                        dnn_run_values[test_dataset_name][train_dataset] = []

                target_value = test_dataset_results["average f1"]

                logger.log_metric(f"dnn zero shot {test_dataset_name}", i, target_value)
                dnn_run_values[test_dataset_name][dataset].append(target_value)

    # Make sure that we have zero-shot test results for the same datasets in both DNN and classical results
    assert all([x in dnn_run_values.keys() for x in classical_run_values.keys()]) and all([x in dnn_run_values.keys() for x in classical_binary_run_values.keys()]), f"Unaligned test datasets detected. DNN:\n{dnn_run_values.keys()}\n\nClassical:\n{classical_run_values.keys()}\n\nClassical binary:\n{classical_binary_run_values.keys()}"


    zero_shot_results = []
    # Get all the results of zero-shot
    for test_task_name in dnn_run_values.keys():

        classical_zero_shot_vals = []
        classical_bin_zero_shot_vals = []
        dnn_zero_shot_vals = []
        classical_in_domain_vals = []
        classical_bin_in_domain_vals = []
        dnn_in_domain_vals = []

        # Get a long list of all the target values for every run of every training set
        for train_task_name in dnn_run_values[test_task_name].keys():
            if train_task_name == test_task_name:
                classical_in_domain_vals = classical_run_values[test_task_name][train_task_name]
                classical_bin_in_domain_vals =  classical_binary_run_values[test_task_name][train_task_name]
                dnn_in_domain_vals = dnn_run_values[test_task_name][train_task_name]
            else:
                classical_zero_shot_vals.extend(classical_run_values[test_task_name][train_task_name])
                classical_bin_zero_shot_vals.extend(classical_binary_run_values[test_task_name][train_task_name])
                dnn_zero_shot_vals.extend(dnn_run_values[test_task_name][train_task_name])

        classical_zs_avg, classical_zs_sd, dnn_zs_avg, dnn_zs_sd, zs_ttest_p_val, zs_wilcoxon_p_val = get_stats_for_two_lists(classical_zero_shot_vals, dnn_zero_shot_vals)
        classical_id_avg, classical_id_sd, dnn_id_avg, dnn_id_sd, id_ttest_p_val, id_wilcoxon_p_val = get_stats_for_two_lists(classical_in_domain_vals, dnn_in_domain_vals)
        classical_bin_zs_avg, classical_bin_zs_sd, _, _, zs_bin_ttest_p_val, zs_bin_wilcoxon_p_val = get_stats_for_two_lists(classical_bin_zero_shot_vals, dnn_zero_shot_vals)
        classical_bin_id_avg, classical_bin_id_sd, _, _, id_bin_ttest_p_val, id_bin_wilcoxon_p_val = get_stats_for_two_lists(classical_bin_in_domain_vals, dnn_in_domain_vals)

        zero_shot_results.append({
            "Classical zero-shot average F1": classical_zs_avg,
            "DNN zero-shot average F1": dnn_zs_avg,
            "Classical zero-shot average F1 stdev": classical_zs_sd,
            "DNN zero-shot average F1 stdev": dnn_zs_sd,
            "Zero-shot T-test p val": zs_ttest_p_val,
            "Zero-shot Wilcoxon p val": zs_wilcoxon_p_val,
            "Classical in-domain average F1": classical_zs_avg,
            "DNN in-domain average F1": dnn_zs_avg,
            "Classical in-domain average F1 stdev": classical_zs_sd,
            "DNN in-domain average F1 stdev": dnn_zs_sd,
            "In-domain T-test p val": zs_ttest_p_val,
            "In-domain Wilcoxon p val": zs_wilcoxon_p_val,
            "Classical binary in-domain average F1": classical_bin_id_avg,
            "Classical binary in-domain average F1 stdev": classical_bin_id_sd,
            "Classical binary zero-shot average F1": classical_bin_zs_avg,
            "Classical binary zero-shot average F1 stdev": classical_bin_zs_sd
        })

    zero_shot_results_df = pd.DataFrame(zero_shot_results, index=dnn_run_values.keys())

    dnn_all_zero_shot_results = dnn_run_values.applymap(lambda x: statistics.mean(x))
    classical_all_zero_shot_results = classical_run_values.applymap(lambda x: statistics.mean(x))

    return zero_shot_results_df, dnn_all_zero_shot_results, classical_all_zero_shot_results
