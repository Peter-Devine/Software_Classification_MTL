from df_to_pytorch_dataset import get_dataset_from_df, get_multiclass_dataset_from_df
from models import get_cls_model_and_optimizer
import torch

class Task:
    def __init__(self, data_getter_fn, is_multilabel):
        self.data_getter_fn = data_getter_fn
        self.is_multilabel = is_multilabel

    def build_task(self, language_model, PARAMS):
        # Download the dataset into a split of train, validation and test Pandas dataframes
        self.dataset = self.data_getter_fn()
        train, valid, test = self.dataset

        # Convert these dataframes into tensor datasets, with inputs (token ids) and labels (integers for multi-class, one-hot vectors for multi-label), as well as the mappings of these values to real labels
        self.train_data, self.valid_data, self.test_data, self.code_map = self.get_tensor_dataset(train, valid, test, PARAMS)

        # Find the number of classes in the dataset
        n_classes = len(code_map.keys())

        # Create a model using the shared language model layer and initialize an optimizer to use with this model
        self.model, self.optimizer = get_cls_model_and_optimizer(language_model, n_classes, PARAMS)

        # Get the loss function that is appropriate for this task
        self.loss_fn = self.get_loss_function()

        # Return this Task object for use in training
        return self

    def get_tensor_dataset(self, train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS):
        if self.is_multilabel:
            tensors_and_map = get_multilabel_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)
        else:
            tensors_and_map = get_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)

        return tensors_and_map

    def get_loss_function(self):
        if self.is_multilabel:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
