import argparse
from task_builder import TaskBuilder
from data_processor_cv import get_cross_validated_df
#from baselines import get_NB_baseline_eval_metrics
from parameters import Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_list', required=True, type=str, help='Comma separated list of datasets (E.g. "maalej_2015,chen_2014"). No spaces between datasets.')
parser.add_argument('--model_name', default="bert-base-uncased", type=str, help='Name of the language model to use (See https://huggingface.co/transformers/pretrained_models.html for all possible models)')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sequence length for input')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size_train', default=20, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size_eval', default=20, type=int, help='Number of epochs to train the model for')
parser.add_argument('--num_fine_tuning_epochs', default=0, type=int, help='Number of epochs to fine tune the model for')
parser.add_argument('--early_stopping_patience', default=999, type=int, help='How many epochs to wait before stopping training after validation performance peak')
parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')
parser.add_argument('--EPS', default=1e-6, type=int, help='Epsilon of the model')
parser.add_argument('--WD', default=0.01, type=int, help='Weight decay of the model')
parser.add_argument('--random_state', default=42, type=int, help='Random state of the experiment (default 42)')
args = parser.parse_args()
print(args)

dataset_list = args.dataset_list.split(",")
PARAMS = Parameters(lm_model_name = args.model_name,
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

if args.do_cv and len(dataset_list) > 1:
    raise Exception("Cannot do literature comparison in multi-task setting")

task_builder = TaskBuilder(random_state=PARAMS.random_state)

task_dict = task_builder.build_tasks(dataset_list, PARAMS)

for task_name, task in task_dict:
    train_df, valid_df, test_df = dataset.dataset


    dataset_name = selected_datasets.keys()[0]

    dataset = selected_datasets[dataset_name]

    cv_datasets = get_cross_validated_df(dataset, PARAMS.cv_folds, PARAMS.random_state)

    model = get_model[dataset_name] ###TODO

    cv_results = []

    for cv_dataset in cv_datasets:
        create_task(dataset_name, dataset_name)
        pass
else:

### THEN ITERATE OVER CVs if applicable

### ELSE JUST DO TRAINING ON GIVEN TVT SPLIT
