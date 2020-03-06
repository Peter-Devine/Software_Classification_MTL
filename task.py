from df_to_pytorch_dataset import get_multilabel_dataset_from_df, get_multiclass_dataset_from_df
from models import get_cls_model_and_optimizer
from torch import nn
from math import ceil

class Task:
    def __init__(self, data_getter_fn, is_multilabel):
        self.data_getter_fn = data_getter_fn
        self.is_multilabel = is_multilabel

    def build_task(self, language_model, PARAMS):
        # Download the dataset into a split of train, validation and test Pandas dataframes
        train, valid, test = self.data_getter_fn()

        # Get the number of training batches in this dataset so that we know how to shuffle this data with respect to others later on
        # E.g. if dataset A has 10 batches and dataset B has 100 batches, we want to train our model on dataset B 10 times more frequently as dataset A
        self.train_length = ceil(train.shape[0] / PARAMS.batch_size_train)

        # Convert these dataframes into tensor datasets, with inputs (token ids) and labels (integers for multi-class, one-hot vectors for multi-label), as well as the mappings of these values to real labels. We also scrape a bunch of useful data of the datasets to compare them in later MTL tasks
        self.train_data, self.valid_data, self.test_data, self.label_map, self.data_info = self.get_tensor_dataset(train, valid, test, PARAMS)

        # Make a training_iterable variable whereupon the dataset can be iterated over, not necessarily in a loop. This is needed for multi-task learning where batches of different tasks will generally be mixed.
        self.training_iterable = iter(self.train_data)

        # Find the number of classes in the dataset
        self.n_classes = len(self.label_map.keys())

        # Create a model using the shared language model layer and initialize an optimizer to use with this model
        self.model, self.optimizer = get_cls_model_and_optimizer(language_model, self.n_classes, PARAMS)

        # Get the loss function that is appropriate for this task
        self.loss_fn = self.get_loss_function()

        # Return this Task object for use in training
        return self

    def get_tensor_dataset(self, train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS):
        if self.is_multilabel:
            tensors_and_map = get_multilabel_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)
        else:
            tensors_and_map = get_multiclass_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)

        return tensors_and_map

    def get_loss_function(self):
        if self.is_multilabel:
            return nn.BCEWithLogitsLoss
        else:
            return nn.CrossEntropyLoss
