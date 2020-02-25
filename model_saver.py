import torch
import os

class ModelSaver:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.get_full_model_file_name = lambda x: f"best_model_{x}.pt"

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_model(self, file_name, model):
        torch.save(model.state_dict(), os.path.join(self.model_dir, self.get_full_model_file_name(file_name)))

    def load_model(self, file_name, model):
        model.load_state_dict(torch.load(os.path.join(self.model_dir, self.get_full_model_file_name(file_name))))
