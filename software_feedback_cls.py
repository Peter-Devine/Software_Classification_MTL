import argparse
from task_builder import TaskBuilder
#from baselines import get_NB_baseline_eval_metrics
from parameters import Parameters
from ml_training import train_on_tasks
from logger import NeptuneLogger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_list', required=True, type=str, help='Comma separated list of datasets (E.g. "maalej_2015,chen_2014"). No spaces between datasets.')
parser.add_argument('--model_name', default="bert-base-uncased", type=str, help='Name of the language model to use (See https://huggingface.co/transformers/pretrained_models.html for all possible models)')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sequence length for input')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size_train', default=64, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size_eval', default=32, type=int, help='Number of epochs to train the model for')
parser.add_argument('--num_fine_tuning_epochs', default=20, type=int, help='Number of epochs to fine tune the model for')
parser.add_argument('--early_stopping_patience', default=999, type=int, help='How many epochs to wait before stopping training after validation performance peak')
parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')
parser.add_argument('--EPS', default=1e-6, type=int, help='Epsilon of the model')
parser.add_argument('--WD', default=0.01, type=int, help='Weight decay of the model')
parser.add_argument('--random_state', default=42, type=int, help='Random state of the experiment (default 42)')
parser.add_argument('--neptune_username', default="", type=str, help=' (Optional) For outputting training/eval metrics to neptune.ai. Valid neptune username. Not your neptune.ai API key must also be stored as $NEPTUNE_API_TOKEN environment variable.')

args = parser.parse_args()
print(args)

dataset_list = args.dataset_list.split(",")
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
                    random_state = args.random_state)

logger = NeptuneLogger(args.neptune_username)
logger.create_experiment(PARAMS)

task_builder = TaskBuilder(random_state=PARAMS.random_state)

task_dict = task_builder.build_tasks(dataset_list, PARAMS)

for task_name, task in task_dict.items():
    logger.log_label_map(task_name, task.code_map)

#Do multi-task learning if more than one task is supplied
if len(dataset_list) > 1:
    task_eval_metrics, task_test_metrics = train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning=False)
    # Output final results to disk
    with open("./task_eval_metrics.txt","w") as f:
        f.write( str(task_eval_metrics) )
    with open("./task_test_metrics.txt","w") as f:
        f.write( str(task_test_metrics) )

# Fine tune on each task individually
ft_task_eval_metrics, ft_task_test_metrics = train_on_tasks(task_dict, PARAMS, logger, is_fine_tuning=True)

# Output final results to disk
with open("./ft_task_eval_metrics.txt","w") as f:
    f.write( str(ft_task_eval_metrics) )
with open("./ft_task_test_metrics.txt","w") as f:
    f.write( str(ft_task_test_metrics) )
