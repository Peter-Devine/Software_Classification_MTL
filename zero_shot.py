from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix
from ignite.metrics import Metric

from evaluation_engines import TopK

class LMZeroShot:
    def __init__(self):
        pass


    # Here, we supply our model which we want to generate predictions on the test inputs from a different dataset to the one the model was trained on
    # We supply the model_mapping, which shows which integers line up with which labels in the model output
    # We also find which integers align with which labels in the label encodings
    def create_zero_shot_eval_engine(self, model, zero_shot_label, model_mapping, label_mapping, cpu):

        model_target_int = [int for label, int in model_mapping.items() if zero_shot_label in label.lower()]
        label_target_int = [int for label, int in label_mapping.items() if zero_shot_label in label.lower()]

        assert len(model_target_int) == 1, f"Ambiguous or empty model label list when trying to map {zero_shot_label} to {model_target_int}"
        assert len(label_target_int) == 1, f"Ambiguous or empty gold label list when trying to map {zero_shot_label} to {model_target_int}"

        model_target_int = model_target_int[0]
        label_target_int = label_target_int[0]

        def process_function(engine, batch):
            X, y = batch
            # pred = model(X.cuda())
            # gold = y.cuda()
            if cpu:
                pred = model(X.cpu())
                gold = y.cpu()
            else:
                pred = model(X.cuda())
                gold = y.cuda()

            pred = pred == model_target_int
            gold = gold == label_target_int

            return pred, gold

        eval_engine = Engine(process_function)

        accuracy = Accuracy()
        accuracy.attach(eval_engine, "accuracy")
        recall = Recall()
        recall.attach(eval_engine, "recall")
        precision = Precision()
        precision.attach(eval_engine, "precision")
        confusion_matrix = ConfusionMatrix(num_classes=2)
        confusion_matrix.attach(eval_engine, "confusion_matrix")
        f1 = (precision * recall * 2 / (precision + recall))
        f1.attach(eval_engine, "f1")
        f2 = (precision * recall * 5 / ((4*precision) + recall))
        f2.attach(eval_engine, "f2")

        avg_recall = Recall(average=True)
        avg_recall.attach(eval_engine, "average recall")
        avg_precision = Precision(average=True)
        avg_precision.attach(eval_engine, "average precision")
        avg_f1 = (avg_precision * avg_recall * 2 / (avg_precision + avg_recall))
        avg_f1.attach(eval_engine, "average f1")
        avg_f2 = (precision * recall * 5 / ((4*precision) + recall))
        avg_f2.attach(eval_engine, "average f2")

        top_k = TopK(k=10, label_idx_of_interest=0)
        top_k.attach(eval_engine, "top_k")

        return eval_engine

    def run_zero_shot_eval(self, task_dict, PARAMS):
        lm_zero_shot_results = {}

        #Iterate over all tasks, and then iterate over each task within each task to compare zero-shot performance across each.
        for task_name, task in task_dict.items():
            for test_task_name, test_task in task_dict.items():
                zero_shot_eval_engine = self.create_zero_shot_eval_engine(task.model, PARAMS.zero_shot_label, task.label_map, test_task.label_map, PARAMS.cpu)

                lm_zero_shot_results[task_name][test_task_name] = zero_shot_eval_engine.run(test_task.test_data).metrics

        return lm_zero_shot_results
