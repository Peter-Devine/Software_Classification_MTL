import argparse
from data_getter import DataGetter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_list', required=True, type=str, help='Comma separated list of datasets (E.g. "maalej_2015,chen_2014"). No spaces between datasets.')
parser.add_argument('--model_name', default="bert-base-uncased", type=str, help='Name of the language model to use (See https://huggingface.co/transformers/pretrained_models.html for all possible models)')
parser.add_argument('--max_length', default=128, type=int, help='Maximum sequence length for input')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train the model for')
parser.add_argument('--num_fine_tuning_epochs', default=0, type=int, help='Number of epochs to fine tune the model for')
parser.add_argument('--early_stopping_patience', default=999, type=int, help='How many epochs to wait before stopping training after validation performance peak')
parser.add_argument('--LR', default=5e-5, type=int, help='Learning rate for the model')
parser.add_argument('--EPS', default=1e-6, type=int, help='Epsilon of the model')
parser.add_argument('--WD', default=0.01, type=int, help='Weight decay of the model')
parser.add_argument('--cv_folds', default=10, type=int, help='Weight decay of the model')
parser.add_argument('--do_baseline',  nargs='?', const=True, default=False, type=bool, help='Get the baseline for the given dataset too.')
parser.add_argument('--do_cv',  nargs='?', const=True, default=False, type=bool, help='Do cross validation evaluation.')
args = parser.parse_args()
print(args)

dataset_list = args.dataset_list.split(",")

if args.do_cv and len(dataset_list) > 1:
    raise Exception("Cannot do cross validation in multi-task setting")

data_getter = DataGetter()

selected_datasets = data_getter.get_selected_datasets(dataset_list)

if args.do_cv:
    

### DO BASELINE TRAINING OVER ALL CVs first

### THEN ITERATE OVER CVs if applicable

### ELSE JUST DO TRAINING ON GIVEN TVT SPLIT
