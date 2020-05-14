import argparse
from task_builder import TaskBuilder
from parameters import Parameters
from ml_training import train_on_tasks
from logger import NeptuneLogger
from baseline import BaselineModels
from zero_shot import LMZeroShot
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_list', required=True, type=str, help='Comma separated list of datasets (E.g. "maalej_2016,chen_2014_swiftkey,ciurumelea_2017_fine"). No spaces between datasets.')
parser.add_argument('--zero_shot_dataset_list', default="", type=str, help='Comma separated list of datasets on which to do zero-shot eval (E.g. "maalej_2016,chen_2014_swiftkey,ciurumelea_2017_fine"). No spaces between datasets.')
parser.add_argument('--model_name', default="bert-base-uncased", type=str, help='Name of the language model to use (See https://huggingface.co/transformers/pretrained_models.html for all possible models)')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sequence length for input')
parser.add_argument('--num_epochs', default=40, type=int, help='Number of epochs to train the model for')
parser.add_argument('--num_fine_tuning_epochs', default=40, type=int, help='Number of epochs to fine tune the model for')
parser.add_argument('--batch_size_train', default=64, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size_eval', default=32, type=int, help='Number of epochs to train the model for')
parser.add_argument('--early_stopping_patience', default=20, type=int, help='How many epochs to wait before stopping training after validation performance peak')
parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')
parser.add_argument('--EPS', default=1e-6, type=int, help='Epsilon of the model')
parser.add_argument('--WD', default=0.01, type=int, help='Weight decay of the model')
parser.add_argument('--best_metric', default="average f1", type=str, help='What metric should be evaluated against for MTL/baseline performance?')
parser.add_argument('--zero_shot_label', default="", type=str, help='What label should the zero-shot comparison be made against?')
parser.add_argument('--random_state', default=42, type=int, help='Random state of the experiment (default 42)')
parser.add_argument('--output_text', type=bool, nargs='?', const=True, default=False, help="Outputs text of the experiment results")
parser.add_argument('--cpu', type=bool, nargs='?', const=True, default=False, help="Uses CPU for processing")
parser.add_argument('--do_classical', type=bool, nargs='?', const=True, default=False, help="Train classical models for comparison?")
parser.add_argument('--neptune_username', default="", type=str, help=' (Optional) For outputting training/eval metrics to neptune.ai. Valid neptune username. Not your neptune.ai API key must also be stored as $NEPTUNE_API_TOKEN environment variable.')

args = parser.parse_args()
print(f"Inputted args are:\n{args}")
dataset_list = args.dataset_list.split(",")
test_dataset_list = args.zero_shot_dataset_list.split(",") if len(args.zero_shot_dataset_list)>0 else []

PARAMS = Parameters(dataset_name_list = dataset_list,
                    lm_model_name = args.model_name,
                    max_length = args.max_length,
                    batch_size_train = args.batch_size_train,
                    batch_size_eval = args.batch_size_eval,
                    learning_rate = args.LR,
                    epsilon = args.EPS,
                    weight_decay = args.WD,
                    early_stopping_patience = args.early_stopping_patience,
                    num_epochs = args.num_epochs,
                    num_fine_tuning_epochs = args.num_fine_tuning_epochs,
                    best_metric = args.best_metric,
                    zero_shot_label = args.zero_shot_label,
                    random_state = args.random_state,
                    cpu= args.cpu)

logger = NeptuneLogger(args.neptune_username)
logger.create_experiment(PARAMS)

task_builder = TaskBuilder(random_state=PARAMS.random_state)

task_dict = task_builder.build_tasks(dataset_list, PARAMS)

# Create the test_task_dict for zero-shot evaluation if we have been supplied test tasks
if len(test_dataset_list) > 0:
    # If we already have a task created in the task_dict, it makes sense to just copy that task into the test_task_dict instead of creating a new task (saves on memory)
    already_created_tasks = [x for x in test_dataset_list if x in task_dict.keys()]
    not_already_created_tasks = [x for x in test_dataset_list if x not in task_dict.keys()]
    test_task_dict = task_builder.build_tasks(not_already_created_tasks, PARAMS, is_test_tasks=True)
    copied_test_tasks = {test_task_name: task_dict[test_task_name] for test_task_name in already_created_tasks}
    test_task_dict.update(copied_test_tasks)

    # Log some useful data that is pertinent when reviewing results of zero-shot evaluation
    for test_task_name, test_task in test_task_dict.items():
        logger.log_dict("label map", test_task.label_map, test_task_name)
        logger.log_dict("task metadata", test_task.data_info, test_task_name)
else:
    test_task_dict = None

# Log some useful data that is pertinent when reviewing results of training / evaluation
for task_name, task in task_dict.items():
    # Only log if the task has not been logged when iterating over the test_task_dict
    if test_task_dict is None or task_name not in test_task_dict.keys():
        logger.log_dict("label map", task.label_map, task_name)
        logger.log_dict("task metadata", task.data_info, task_name)

if args.do_classical:
    # Do per-task in-domain training and evaluation first
    baseline_models = BaselineModels(random_state=PARAMS.random_state)
    for task_name, task in task_dict.items():
        best_classical_result, all_classical_results = baseline_models.get_baselines(task.train_df, task.valid_df, task.test_df, best_metric=PARAMS.best_metric, is_multilabel=task.is_multilabel)
        logger.log_dict("best baselines", best_classical_result, task_name)
        if args.output_text:
            dataset_string = "__".join(dataset_list)
            logger.log_json(f"{dataset_string}_{PARAMS.random_state}_best_classical_baselines.json", best_classical_result)

        logger.log_dict("all baselines", all_classical_results, task_name)

    # Run classical zero-shot learning on all datasets if we have a designated set of test tasks, the run out of domain (zero shot) evaluation on classical models
    if len(PARAMS.zero_shot_label) > 0 and len(test_task_dict.keys()) > 0:
        zero_shot_results = baseline_models.get_zero_shot_baselines(task_dict, test_task_dict, PARAMS.best_metric, PARAMS.zero_shot_label)
        mtl_zero_shot_results = baseline_models.get_MTL_baselines(task_dict, test_task_dict, PARAMS.best_metric, PARAMS.zero_shot_label)
        logger.log_dict("Zero shot results (classical)", zero_shot_results)
        logger.log_dict("MTL zero shot results (classical)", mtl_zero_shot_results)

        if args.output_text:
            dataset_string = "__".join(dataset_list)
            logger.log_json(f"{dataset_string}_{PARAMS.random_state}_zero_shot_classical_baselines.json", zero_shot_results)
            logger.log_json(f"{dataset_string}_{PARAMS.random_state}_mtl_zero_shot_classical_baselines.json", mtl_zero_shot_results)

        del zero_shot_results, mtl_zero_shot_results

    del baseline_models, best_classical_result, all_classical_results

# Do multi-task learning if more than one task is supplied
if len(dataset_list) > 1:
    task_eval_metrics, task_test_metrics = train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning=False)

    if args.output_text:
        # Output final results to disk
        dataset_string = "__".join(dataset_list)
        logger.log_json(f"{dataset_string}_{PARAMS.random_state}_task_eval_metrics.json", task_eval_metrics)
        logger.log_json(f"{dataset_string}_{PARAMS.random_state}_task_test_metrics.json", task_test_metrics)

# Fine tune on each task individually
ft_task_eval_metrics, ft_task_test_metrics = train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning=True)

if args.output_text:
    # Output final results to disk
    dataset_string = "__".join(dataset_list)
    logger.log_json(f"{dataset_string}_{PARAMS.random_state}_ft_task_eval_metrics.json", ft_task_eval_metrics)
    logger.log_json(f"{dataset_string}_{PARAMS.random_state}_ft_task_test_metrics.json", ft_task_test_metrics)

# If we have test tasks and a label with which to compare them, then we run zero-shot evaluation
if len(PARAMS.zero_shot_label) > 0 and len(test_task_dict.keys()) > 0:
    lm_zero_shot = LMZeroShot()
    lm_zero_shot_results = lm_zero_shot.run_zero_shot_eval(task_dict, test_task_dict, PARAMS)
    logger.log_dict("LM zero shot", lm_zero_shot_results)
    if args.output_text:
        dataset_string = "__".join(dataset_list)
        logger.log_json(f"{dataset_string}_{PARAMS.random_state}_zero_shot_test_metrics.json", lm_zero_shot_results)

logger.stop()
