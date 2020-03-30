import torch
import os

class ModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.get_full_model_file_name = lambda x: f"best_model_{x}.pt"

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_model(self, file_name, model):
        model_binary_path = os.path.join(self.model_dir, self.get_full_model_file_name(file_name))
        torch.save(model.state_dict(), model_binary_path)

    def load_model(self, file_name, model):
        model_binary_path = os.path.join(self.model_dir, self.get_full_model_file_name(file_name))
        model.load_state_dict(torch.load(model_binary_path))
