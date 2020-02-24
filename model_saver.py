import torch
import os

class ModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.get_full_model_file_name = lambda x: f"best_model_{x}.py"

    def save_model(self, file_name, model):
        torch.save(model, os.path.join(self.model_dir, self.get_full_model_file_name(file_name)))

    def load_model(self, file_name):
        return torch.load(os.path.join(self.model_dir, self.get_full_model_file_name(file_name)))
