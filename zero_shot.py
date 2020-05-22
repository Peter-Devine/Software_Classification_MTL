import torch

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

        # Iterate through all labels in both the train and test sets to see which labels correspond to the zero shot label (the unifying label)
        model_target_int = [int for label, int in model_mapping.items() if zero_shot_label in label.lower()]
        label_target_int = [int for label, int in label_mapping.items() if zero_shot_label in label.lower()]

        # There should only be one unifying label in each dataset (Possible TODO: Allow multiple labels to map to one unifying label)
        assert len(model_target_int) == 1, f"Ambiguous or empty model label list when trying to map {zero_shot_label} to {model_target_int}"
        assert len(label_target_int) == 1, f"Ambiguous or empty gold label list when trying to map {zero_shot_label} to {model_target_int}"

        model_target_int = model_target_int[0]
        label_target_int = label_target_int[0]

        def process_function(engine, batch):
            X, y = batch

            if cpu:
                pred = model(X.cpu())
                gold = y.cpu()
            else:
                pred = model(X.cuda())
                gold = y.cuda()

            # Get the softmax of the raw model output (logits)
            pred = torch.softmax(pred, dim=1)

            # Get the probability that the prediction is the target class
            pred_in_class_prob = pred[:,[model_target_int]]

            # Get all the probabilities of all the other classes outside the target class by finding the complement of the in class probability
            pred_out_class_prob = 1 - pred_in_class_prob

            # Create a combined tensor which acts as a set of probabilities for in vs out of the zero-shot target class.
            # In this, 0 is out of class, whilst 1 is in class, so the combined tensor has the out of class probabilities in the 0th column and the in-class probs in the 1st column.
            pred = torch.cat((pred_out_class_prob, pred_in_class_prob), dim=1)

            # To correspond to the above contructed tensor, we set the golds as 1 (I.e. True) if the gold label is the zero-shot label, and 0 (False) if not.
            gold = (gold == label_target_int).long()

            return pred, gold

        eval_engine = Engine(process_function)

        really_small_number = 1e-10

        accuracy = Accuracy()
        accuracy.attach(eval_engine, "accuracy")
        recall = Recall()
        recall.attach(eval_engine, "recall")
        precision = Precision()
        precision.attach(eval_engine, "precision")
        f1 = (precision * recall * 2 / (precision + recall + really_small_number))
        f1.attach(eval_engine, "f1")
        f2 = (precision * recall * 5 / ((4*precision) + recall + really_small_number))
        f2.attach(eval_engine, "f2")

        avg_recall = Recall(average=True)
        avg_recall.attach(eval_engine, "average recall")
        avg_precision = Precision(average=True)
        avg_precision.attach(eval_engine, "average precision")
        avg_f1 = (avg_precision * avg_recall * 2 / (avg_precision + avg_recall + really_small_number))
        avg_f1.attach(eval_engine, "average f1")
        avg_f2 = (avg_precision * avg_recall * 5 / ((4*avg_precision) + avg_recall + really_small_number))
        avg_f2.attach(eval_engine, "average f2")

        return eval_engine

    def run_zero_shot_eval(self, task_dict, test_task_dict, PARAMS):
        lm_zero_shot_results = {}

        #Iterate over all tasks, and then iterate over each task within each task to compare zero-shot performance across each.
        for task_name, task in task_dict.items():
            lm_zero_shot_results[task_name] = {}
            for test_task_name, test_task in test_task_dict.items():
                zero_shot_eval_engine = self.create_zero_shot_eval_engine(task.model, PARAMS.zero_shot_label, task.label_map, test_task.label_map, PARAMS.cpu)

                lm_zero_shot_results[task_name][test_task_name] = zero_shot_eval_engine.run(test_task.test_data).metrics

        return lm_zero_shot_results
