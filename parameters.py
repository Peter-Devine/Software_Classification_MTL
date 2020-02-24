class Parameters:
    def __init__(self,
                dataset_name_list,
                lm_model_name,
                max_length,
                batch_size_train,
                batch_size_eval,
                learning_rate,
                epsilon,
                weight_decay,
                early_stopping_patience,
                num_epochs,
                num_fine_tuning_epochs,
                random_state):
        self.dataset_name_list = dataset_name_list
        self.lm_model_name = lm_model_name
        self.max_length = max_length
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.num_epochs = num_epochs
        self.num_fine_tuning_epochs = num_fine_tuning_epochs
        self.random_state = random_state
