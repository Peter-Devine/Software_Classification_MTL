from ignite.engine import Engine
from ignite.metrics import Accuracy, Recall, Precision, ConfusionMatrix

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

  if not is_multilabel:
      confusion_matrix = ConfusionMatrix(num_classes=n_classes)
      confusion_matrix.attach(eval_engine, "confusion_matrix")

  F1 = (precision * recall * 2 / (precision + recall))
  F1.attach(eval_engine, "f1")

  return eval_engine
