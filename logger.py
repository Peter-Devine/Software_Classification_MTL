import neptune

class NeptuneLogger:
    def __init__(self, username):
        # HAVE YOUR API KEY SAVED AS AN ENV VAR $NEPTUNE_API_TOKEN (or you can provide it as a "api_token" argument below if your code is private)
        neptune.init(username + '/sandbox')

    def create_experiment(self, PARAMS):
        neptune.create_experiment(name="__|__".join(PARAMS.dataset_name_list),
                          params=vars(PARAMS))

    def log_metric(self, metric_name, x, y):
        neptune.log_metric(metric_name, x, y)

    def is_numeric(self, value):
        try:
            float(value)
            return True
        except Exception:
            return False

    def log_array(self, metric_name, x, array):
        for i, cell in enumerate(array):
            inner_metric_name = f"{metric_name}__{str(i)}"
            if self.is_numeric(cell):
                neptune.log_metric(inner_metric_name, x, cell)
            else:
                self.log_array(inner_metric_name, x, cell)

    def log_text(self, metric_name, x, text):
        neptune.log_text(metric_name, x, text)

    def log_label_map(self, label_map, task_name):
        neptune.log_text(f"{task_name} label map", str(label_map))

    def log_results(self, task_name, split_type, epoch, results_dict):
        metric_prefix = f"{task_name} {split_type} "
        for metric_name, metric in results_dict.items():
            if self.is_numeric(metric):
                self.log_metric(metric_prefix + metric_name, epoch, metric)
            else:
                self.log_text(metric_prefix + metric_name, epoch, str(metric))

    def stop(self):
        neptune.stop()
