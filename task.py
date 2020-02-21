from df_to_pytorch_dataset import get_dataset_from_df, get_multiclass_dataset_from_df
from models import get_cls_model_and_optimizer
import torch

class Task:
    def __init__(self, data_getter_fn, is_multiclass):
        self.data_getter_fn = data_getter_fn
        self.is_multiclass = is_multiclass

    def build_task(self, language_model, PARAMS):
        self.dataset = self.data_getter_fn()
        train, valid, test = self.dataset
        self.train_data, self.valid_data, self.test_data, self.code_map = self.get_tensor_dataset(train, valid, test, PARAMS)
        n_classes = len(code_map.keys())

        self.model, self.optimizer = get_cls_model_and_optimizer(language_model, n_classes, PARAMS)

        self.loss_fn = self.get_loss_function()
        return self

    def get_tensor_dataset(self, train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS):
        if self.is_multiclass:
            tensors_and_map = get_multiclass_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)
        else:
            tensors_and_map = get_dataset_from_df(train_tensor_dataset, valid_tensor_dataset, test_tensor_dataset, PARAMS)

        return tensors_and_map

    def get_loss_function(self):
        if self.is_multiclass:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
