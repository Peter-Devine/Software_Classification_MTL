from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix
from ignite.metrics import Metric
import torch
import numpy as np

class TopK(Metric):
    def __init__(self, output_transform=lambda x: x, k=10, label_idx_of_interest=0):
        self.k = k
        self.label_idx_of_interest = label_idx_of_interest
        self.top_k_values = None
        self.top_k_labels = None
        super(TopK, self).__init__(output_transform=output_transform)

    def reset(self):
        self.top_k_values = None
        self.top_k_labels = None

    def update(self, output):
        y_pred, y = output

        if self.top_k_values is None:
            top_k = torch.topk(torch.softmax(y_pred, dim=1)[:,self.label_idx_of_interest], self.k, dim=0)
            self.top_k_values = top_k.values.cpu()
            self.top_k_labels = y[top_k.indices].cpu()
        else:
            concatenated_values = torch.cat((torch.softmax(y_pred, dim=1)[:,self.label_idx_of_interest].cpu(), self.top_k_values), dim=0)
            concatenated_labels = torch.cat((y.cpu(), self.top_k_labels), dim=0)
            top_k = torch.topk(concatenated_values, self.k, dim=0)
            self.top_k_values = top_k.values
            self.top_k_labels = concatenated_labels[top_k.indices]

    def compute(self):
        assert len(self.top_k_labels) == self.k, f"There are {self.top_k_labels} labels when there should be {self.k} labels when calculating top k labels"
        number_correct = (self.top_k_labels == self.label_idx_of_interest).sum()
        percentage_in_top_k = float(number_correct) / float(self.k)
        return percentage_in_top_k

class CountHolder():
    def __init__(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def update(self, fp, fn, tp, tn):
        self.fp += fp
        self.fn += fn
        self.tp += tp
        self.tn += tn

class MulticlassPrecisionRecall(Metric):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        self.class_count = []
        for i in range(n_classes):
            self.class_count.append(CountHolder())

        self.n_classes = n_classes
        self.sigmoid_fn = torch.nn.Sigmoid()
        super(MulticlassPrecisionRecall, self).__init__(output_transform=output_transform)

    def reset(self):
        for i in range(self.n_classes):
            self.class_count[i] = CountHolder()

    def update(self, output):
        y_pred_all, y_all = output

        for i in range(self.n_classes):
            y_pred = y_pred_all[:, i]
            y_pred = self.sigmoid_fn(y_pred)
            y_pred = y_pred.round()
            y = y_all[:, i]
            fp = int(((y_pred == 1) & (y == 0)).sum())
            fn = int(((y_pred == 0) & (y == 1)).sum())
            tp = int(((y_pred == 1) & (y == 1)).sum())
            tn = int(((y_pred == 0) & (y == 0)).sum())

            self.class_count[i].update(fp=fp, fn=fn, tp=tp, tn=tn)

class MulticlassPrecision(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2, average=False):
        super(MulticlassPrecision, self).__init__(output_transform=output_transform, n_classes=n_classes)
        self.average = average

    def compute(self):
        results = []
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fp = self.class_count[i].fp
            if (tp + fp) == 0:
                results.append(np.nan)
            else:
                results.append(round(tp / (tp + fp), 5))

        if self.average:
            return sum(results) / len(results)
        else:
            return results

class MulticlassRecall(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2, average=False):
        super(MulticlassRecall, self).__init__(output_transform=output_transform, n_classes=n_classes)
        self.average = average

    def compute(self):
        results = []
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fn = self.class_count[i].fn
            if (tp + fn) == 0:
                results.append(np.nan)
            else:
                results.append(round(tp / (tp + fn), 5))

        if self.average:
            return sum(results) / len(results)
        else:
            return results

class MulticlassF(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2, average=False, f_n=1):
        super(MulticlassF, self).__init__(output_transform=output_transform, n_classes=n_classes)
        self.f_n = f_n
        self.average = average

    def compute(self):
        results = []
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fn = self.class_count[i].fn
            fp = self.class_count[i].fp

            prec = tp / (tp + fp) if (tp + fp) != 0 else None
            rec = tp / (tp + fn) if (tp + fn) != 0 else None

            if (prec == 0 and rec == 0) or prec is None or rec is None:
                results.append(np.nan)
            else:
                beta_squared = self.f_n**2
                f_score = (1+beta_squared) * prec * rec / ((beta_squared*prec) + rec)
                results.append(round(f_score, 5))

        if self.average:
            return sum(results) / len(results)
        else:
            return results


class MulticlassAccuracy(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def get_averages(self):
        results = []
        hits = 0
        misses = 0
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fn = self.class_count[i].fn
            fp = self.class_count[i].fp
            tn = self.class_count[i].tn

            hits += tp + tn
            misses += fp + fn

            if (tp + tn + fp + fn) == 0:
                results.append(np.nan)
            else:
                results.append(round((tp + tn) / (tp + tn + fp + fn), 5))

        overall_accuracy = round(hits / (hits + misses), 5) if hits + misses > 0 else np.nan

        return {"per_class": results, "overall": overall_accuracy}

class MulticlassOverallAccuracy(MulticlassAccuracy):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassOverallAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        return self.get_averages()["overall"]

class MulticlassPerClassAccuracy(MulticlassAccuracy):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassPerClassAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        return self.get_averages()["per_class"]

def create_eval_engine(model, is_multilabel, n_classes, cpu):

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

      return pred, gold

  eval_engine = Engine(process_function)

  if is_multilabel:
      accuracy = MulticlassOverallAccuracy(n_classes=n_classes)
      accuracy.attach(eval_engine, "accuracy")
      per_class_accuracy = MulticlassPerClassAccuracy(n_classes=n_classes)
      per_class_accuracy.attach(eval_engine, "per class accuracy")
      recall = MulticlassRecall(n_classes=n_classes)
      recall.attach(eval_engine, "recall")
      precision = MulticlassPrecision(n_classes=n_classes)
      precision.attach(eval_engine, "precision")
      f1 = MulticlassF(n_classes=n_classes, f_n=1)
      f1.attach(eval_engine, "f1")
      f2= MulticlassF(n_classes=n_classes, f_n=2)
      f2.attach(eval_engine, "f2")

      avg_recall = MulticlassRecall(n_classes=n_classes, average=True)
      avg_recall.attach(eval_engine, "average recall")
      avg_precision = MulticlassPrecision(n_classes=n_classes, average=True)
      avg_precision.attach(eval_engine, "average precision")
      avg_f1 = MulticlassF(n_classes=n_classes, average=True, f_n=1)
      avg_f1.attach(eval_engine, "average f1")
      avg_f2= MulticlassF(n_classes=n_classes, average=True, f_n=2)
      avg_f2.attach(eval_engine, "average f2")
  else:
      accuracy = Accuracy()
      accuracy.attach(eval_engine, "accuracy")
      recall = Recall()
      recall.attach(eval_engine, "recall")
      precision = Precision()
      precision.attach(eval_engine, "precision")
      confusion_matrix = ConfusionMatrix(num_classes=n_classes)
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
      avg_f2 = (avg_precision * avg_recall * 5 / ((4*avg_precision) + avg_recall))
      avg_f2.attach(eval_engine, "average f2")

      if n_classes == 2:
          top_k = TopK(k=10, label_idx_of_interest=0)
          top_k.attach(eval_engine, "top_k")

  return eval_engine
