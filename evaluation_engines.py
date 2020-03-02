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
            self.top_k_values = top_k.values
            self.top_k_labels = y[top_k.indices]
        else:
            concatenated_values = torch.cat((torch.softmax(y_pred, dim=1)[:,self.label_idx_of_interest], self.top_k_values), dim=0)
            concatenated_labels = torch.cat((y, self.top_k_labels), dim=0)
            top_k = torch.topk(concatenated_values, self.k, dim=0)
            self.top_k_values = top_k.values
            self.top_k_labels = concatenated_labels[top_k.indices]

    def compute(self):
        assert len(self.top_k_labels) == self.k, f"There are {self.top_k_labels} labels when there should be {self.k} labels when calculating top k labels"
        number_correct = (self.top_k_labels == self.label_idx_of_interest).sum()
        percentage_in_top_k = float(number_correct) / float(self.k)
        return percentage_in_top_k

class MulticlassPrecisionRecall(Metric):
    def __init__(self, output_transform=lambda x: x, label_idx_of_interest=0):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0
        self.label_idx_of_interest = label_idx_of_interest
        self.sigmoid_fn = torch.nn.Sigmoid()
        super(MulticlassPrecisionRecall, self).__init__(output_transform=output_transform)

    def reset(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0

    def update(self, output):
        y_pred, y = output
        y_pred = y_pred[:, self.label_idx_of_interest]
        y_pred = self.sigmoid_fn(y_pred)
        y_pred = y_pred.round()
        y = y[:, self.label_idx_of_interest]
        self.fp += int(((y_pred == 1) & (y == 0)).sum())
        self.fn += int(((y_pred == 0) & (y == 1)).sum())
        self.tp += int(((y_pred == 1) & (y == 1)).sum())
        self.tn += int(((y_pred == 0) & (y == 0)).sum())

class MulticlassPrecision(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, label_idx_of_interest=0):
        super(MulticlassPrecision, self).__init__(output_transform=output_transform, label_idx_of_interest=label_idx_of_interest)

    def compute(self):
        print("MulticlassPrecision")
        if (self.tp + self.fp) == 0:
            return np.nan
        else:
            return self.tp / (self.tp + self.fp)

class MulticlassRecall(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, label_idx_of_interest=0):
        super(MulticlassRecall, self).__init__(output_transform=output_transform, label_idx_of_interest=label_idx_of_interest)

    def compute(self):
        print("MulticlassRecall")
        print((self.tp + self.fn))
        if (self.tp + self.fn) == 0:
            return np.nan
        else:
            return self.tp / (self.tp + self.fn)

class MulticlassSingleClassAccuracy(MulticlassPrecisionRecall):
    def __init__(self, output_transform=lambda x: x, label_idx_of_interest=0):
        super(MulticlassSingleClassAccuracy, self).__init__(output_transform=output_transform, label_idx_of_interest=label_idx_of_interest)

    def compute(self):
        print("MulticlassSingleClassAccuracy")
        if (self.tp + self.tn + self.fp + self.fn) == 0:
            return np.nan
        else:
            return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

class MulticlassAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.correct = 0
        self.all = 0
        self.sigmoid_fn = torch.nn.Sigmoid()
        super(MulticlassAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self.correct = 0
        self.all = 0

    def update(self, output):
        y_pred, y = output
        y_pred = self.sigmoid_fn(y_pred)
        y_pred = y_pred.round()
        self.correct += int((y_pred == y).sum())
        self.all += int(y.shape[0] * y.shape[1])

    def compute(self):
        print("MulticlassAccuracy")
        if self.all == 0:
            return np.nan
        else:
            return self.correct / self.all

def create_eval_engine(model, is_multilabel, n_classes):

  def process_function(engine, batch):
      X, y = batch
      # pred = model(X.cuda())
      # gold = y.cuda()
      pred = model(X.cpu())
      gold = y.cpu()
      return pred, gold

  eval_engine = Engine(process_function)


  if is_multilabel:
      accuracy = MulticlassAccuracy()
      accuracy.attach(eval_engine, "accuracy")
      for i in range(n_classes):
          accuracy = MulticlassSingleClassAccuracy(label_idx_of_interest=i)
          accuracy.attach(eval_engine, f"accuracy{str(n_classes)}")
          recall = MulticlassRecall(label_idx_of_interest=i)
          recall.attach(eval_engine, f"recall{str(n_classes)}")
          precision = MulticlassPrecision(label_idx_of_interest=i)
          precision.attach(eval_engine, f"precision{str(n_classes)}")
          f1 = (precision * recall * 2 / (precision + recall))
          f1.attach(eval_engine, f"f1{str(n_classes)}")
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
