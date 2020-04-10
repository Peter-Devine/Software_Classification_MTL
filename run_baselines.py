# import argparse
# from baseline import BaselineModels
# from task_builder import TaskBuilder
# from logger import NeptuneLogger
# from parameters import Parameters
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_list', required=True, type=str, help='Comma separated list of datasets (E.g. "maalej_2015,chen_2014_swiftkey,ciurumelea_2017_fine"). No spaces between datasets.')
# parser.add_argument('--best_metric', default="average f1", type=str, help='What metric should be evaluated against for MTL/baseline performance?')
# parser.add_argument('--random_state', default=42, type=int, help='Random state of the experiment (default 42)')
# parser.add_argument('--neptune_username', default="", type=str, help=' (Optional) For outputting training/eval metrics to neptune.ai. Valid neptune username. Not your neptune.ai API key must also be stored as $NEPTUNE_API_TOKEN environment variable.')
#
# args = parser.parse_args()
# print(f"Inputted args are:\n{args}")
# dataset_list = args.dataset_list.split(",")
#
# logger = NeptuneLogger(args.neptune_username)
#
# PARAMS = Parameters(dataset_name_list = dataset_list,
#                     lm_model_name = None,
#                     max_length = None,
#                     batch_size_train = None,
#                     batch_size_eval = None,
#                     learning_rate = None,
#                     epsilon = None,
#                     weight_decay = None,
#                     early_stopping_patience = None,
#                     num_epochs = None,
#                     num_fine_tuning_epochs = None,
#                     best_metric = args.best_metric,
#                     random_state = args.random_state,
#                     cpu= None)
#
# logger.create_experiment(PARAMS)
#
# task_builder = TaskBuilder(random_state=args.random_state)
#
#
# for dataset in dataset_list:
#     task = task_builder.task_dict[dataset]
#     train, valid, test = task.data_getter_fn()
#     baseline_models = BaselineModels(is_multilabel=task.is_multilabel)
#     best_baseline_values, all_baseline_values = baseline_models.get_baselines(train, valid, test, args.best_metric)
#
#     logger.log_dict("best baselines", dataset, best_baseline_values)
#     logger.log_dict("all baselines", dataset, all_baseline_values)
#
# logger.stop()
