import neptune

class NeptuneLogger:
    def __init__(self, username):
        if len(username)>0:
            # HAVE YOUR API KEY SAVED AS AN ENV VAR $NEPTUNE_API_TOKEN (or you can provide it as a "api_token" argument below if your code is private)
            neptune.init(username + '/sandbox')
            self.logger_active = True
        else:
            self.logger_active = False

    def create_experiment(self, PARAMS):
        if self.logger_active:
            neptune.create_experiment(name="__|__".join(PARAMS.dataset_name_list),
                              params=vars(PARAMS))

    def log_metric(self, metric_name, x, y):
        if self.logger_active:
            neptune.log_metric(metric_name, x, y)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \nmetric:{str(y)}\n\n")

    def is_numeric(self, value):
        try:
            float(value)
            return True
        except Exception:
            return False

    def log_array(self, metric_name, x, array):
        if self.logger_active:
            for i, cell in enumerate(array):
                inner_metric_name = f"{metric_name}__{str(i)}"
                if self.is_numeric(cell):
                    neptune.log_metric(inner_metric_name, x, cell)
                else:
                    self.log_array(inner_metric_name, x, cell)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \narray:{str(array)}\n\n")

    def log_text(self, metric_name, x, text):
        if self.logger_active:
            neptune.log_text(metric_name, x, text)
        else:
            print(f"metric_name: {metric_name}, \nx:{x}, \ntext:{text}\n\n")

    def log_dict(self, dict_name, task_name, input_dict, recursion_level = 0):
        if self.logger_active:
            # Add spaces so that dict prints prettily in logger
            spacing_str = '  '*recursion_level
            for key, value in input_dict.items():
                if type(value) == dict:
                    neptune.log_text(f"{task_name} {dict_name}", f"{spacing_str}{str(key)}")
                    self.log_dict(dict_name, task_name, value, recursion_level+1)
                else:
                    neptune.log_text(f"{task_name} {dict_name}", f"{spacing_str}{str(key)}: {str(value)}")
        else:
            print(f"{task_name} {dict_name}: {str(dict)}")

    def log_results(self, task_name, split_type, epoch, results_dict):
        if self.logger_active:
            metric_prefix = f"{task_name} {split_type} "
            for metric_name, metric in results_dict.items():
                if self.is_numeric(metric):
                    self.log_metric(metric_prefix + metric_name, epoch, metric)
                else:
                    self.log_text(metric_prefix + metric_name, epoch, str(metric))

    def stop(self):
        if self.logger_active:
            neptune.stop()
