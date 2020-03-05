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
            concatenated_values = torch.cat((torch.softmax(y_pred, dim=1)[:,self.label_idx_of_interest], self.top_k_values), dim=0)
            concatenated_labels = torch.cat((y, self.top_k_labels), dim=0)
            top_k = torch.topk(concatenated_values, self.k, dim=0)
            self.top_k_values = top_k.values.cpu()
            self.top_k_labels = concatenated_labels[top_k.indices].cpu()

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

            for i in range(self.n_classes):
                self.class_count[i].update(fp=fp, fn=fn, tp=tp, tn=tn)

class MulticlassPrecision(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassPrecision, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        results = []
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fp = self.class_count[i].fp
            if (tp + fp) == 0:
                results.append(np.nan)
            else:
                results.append(tp / (tp + fp))
        return results

class MulticlassRecall(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassRecall, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        results = []
        for i in range(self.n_classes):
            tp = self.class_count[i].tp
            fn = self.class_count[i].fn
            if (tp + fn) == 0:
                results.append(np.nan)
            else:
                results.append(tp / (tp + fn))
        return results

class MulticlassF1(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassF1, self).__init__(output_transform=output_transform, n_classes=n_classes)

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
                results.append(prec * rec * 2 / (prec + rec))
        return results


class MulticlassAccuracy(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
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
                results.append((tp + tn) / (tp + tn + fp + fn))

        overall_accuracy = hits / (hits + misses) if hits + misses > 0 else np.nan

        return {"per_class": results, "overall": overall_accuracy}

class MulticlassOverallAccuracy(MulticlassAccuracy):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassOverallAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        return self.compute()["overall"]

class MulticlassPerClassAccuracy(MulticlassAccuracy):
    def __init__(self, output_transform=lambda x: x, n_classes=2):
        super(MulticlassOverallAccuracy, self).__init__(output_transform=output_transform, n_classes=n_classes)

    def compute(self):
        return self.compute()["per_class"]

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
      per_class_accuracy.attach(eval_engine, f"per class accuracy")
      recall = MulticlassRecall(n_classes=n_classes)
      recall.attach(eval_engine, f"recall")
      precision = MulticlassPrecision(n_classes=n_classes)
      precision.attach(eval_engine, f"precision")
      f1 = MulticlassF1(n_classes=n_classes)
      f1.attach(eval_engine, f"f1")
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

      if n_classes == 2:
          top_k = TopK(k=10, label_idx_of_interest=0)
          top_k.attach(eval_engine, "top_k")

  return eval_engine
