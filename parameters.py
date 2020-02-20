class Hyperparameters:
    def __init__(self, lm_model_name, max_length, batch_size_train, batch_size_eval):
        self.lm_model_name = lm_model_name
        self.max_length = max_length
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
