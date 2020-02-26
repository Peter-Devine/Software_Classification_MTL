from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix
from ignite.metrics import Metric
import torch


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

def create_eval_engine(model, is_multilabel, n_classes):

  def process_function(engine, batch):
      X, y = batch
      # pred = model(X.cuda())
      # gold = y.cuda()
      pred = model(X.cpu())
      gold = y.cpu()
      return pred, gold

  eval_engine = Engine(process_function)
  accuracy = Accuracy(is_multilabel=is_multilabel)
  accuracy.attach(eval_engine, "accuracy")
  recall = Recall(is_multilabel=is_multilabel)
  recall.attach(eval_engine, "recall")
  precision = Precision(is_multilabel=is_multilabel)
  precision.attach(eval_engine, "precision")
  top_k = TopK(k=10, label_idx_of_interest=0)
  top_k.attach(eval_engine, "top_k")

  if not is_multilabel:
      confusion_matrix = ConfusionMatrix(num_classes=n_classes)
      confusion_matrix.attach(eval_engine, "confusion_matrix")

  F1 = (precision * recall * 2 / (precision + recall))
  F1.attach(eval_engine, "f1")

  return eval_engine
