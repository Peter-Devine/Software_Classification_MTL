from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from functools import partial
import time

class BaselineModels:
    def __init__(self):
        self.model_dict = {
            "Gaussian NB": GaussianNB,
            "Complement NB": ComplementNB,
            "Bernoulli NB": BernoulliNB,
            "Multinomial NB": MultinomialNB,
            "Decision tree classifier": DecisionTreeClassifier,
            "Support vector classifier": SVC,
            "Gradient boosting classifier": GradientBoostingClassifier
        }
        self.metrics = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "average f1": partial(f1_score, average="macro"),
            "precision": precision_score,
            "recall": recall_score
        }

    def get_zero_shot_baselines(self, task_dict, best_metric, zero_shot_label):
        zero_shot_results = {}
        for task_name, task in task_dict.items():
            zero_shot_results[task_name] = {}
            for test_task_name, test_task in task_dict.items():
                if task.is_multilabel:
                    label_column_names = [x for x in task.columns if zero_shot_label in x.lower()]
                    assert len(label_column_names) == 1, f"Not exactly one column name that corresponds to zero shot label {zero_shot_label} ({task.columns})"
                    label_column_names = label_column_names[0]
                    train_df_labels = task.train_df[label_column_name].apply(lambda x: True if x else False)
                    valid_df_labels = task.valid_df[label_column_name].apply(lambda x: True if x else False)
                else:
                    assert any([zero_shot_label in x.lower() for x in task.train_df.labels.unique()]), f"Zero shot label, {zero_shot_label}, not in dataset (Vals: {task.train_df.labels.unique()})"
                    train_df_labels = task.train_df.labels.apply(lambda x: True if zero_shot_label in x.lower() else False)
                    valid_df_labels = task.valid_df.labels.apply(lambda x: True if zero_shot_label in x.lower() else False)

                if test_task.is_multilabel:
                    label_column_name = [x for x in test_task.test_df.columns if zero_shot_label in x.lower()][0]
                    test_df_labels = test_task.test_df[label_column_name].apply(lambda x: True if x else False)
                else:
                    test_df_labels = test_task.test_df.labels.apply(lambda x: True if zero_shot_label in x.lower() else False)

                train_df = pd.DataFrame({"text": task.train_df.text, "baseline_label": train_df_labels})
                valid_df = pd.DataFrame({"text": task.valid_df.text, "baseline_label": valid_df_labels})
                test_df = pd.DataFrame({"text": test_task.test_df.text, "baseline_label": test_df_labels})

                best_results, results = get_results_on_binary_task(train_df, valid_df, test_df, is_multilabel=False, best_metric=best_metric)
                zero_shot_results[task_name][test_task_name] = best_results

        return zero_shot_results

    def get_results_on_binary_task(self, train_df, valid_df, test_df, best_metric, multiclass=False):

        bow_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfTransformer()

        def transform_text(train, valid, test, is_idf):
            train_feat = bow_vectorizer.fit_transform(train.text).toarray()
            valid_feat = bow_vectorizer.transform(valid.text).toarray()
            test_feat = bow_vectorizer.transform(test.text).toarray()

            if is_idf:
                train_feat = tfidf_vectorizer.fit_transform(train_feat).toarray()
                valid_feat = tfidf_vectorizer.transform(valid_feat).toarray()
                test_feat = tfidf_vectorizer.transform(test_feat).toarray()

            return train_feat, valid_feat, test_feat

        bow_splits = transform_text(train_df, valid_df, test_df, is_idf=False)
        tfidf_splits = transform_text(train_df, valid_df, test_df, is_idf=True)

        input_types = {
            "bow": bow_splits,
            "tfidf": tfidf_splits
        }

        results = {}

        # Save the best metric, model and configuration (model name and input type) based on the validation score
        best_score = None
        best_model = None
        best_config = None
        best_results = {}

        # If we are doing a multiclass baseline, then we need to know which labels are referred to by which columns in our prec, rec, f1 output
        # We fit an sklearn binarizer on the training data and output the mappings of these classes to our results output
        if multiclass:
            binarizer = LabelBinarizer()
            binarizer.fit(train_df.baseline_label)
            results["multiclass label map"] = binarizer.classes_
            best_results["multiclass label map"] = binarizer.classes_

        # Start a clock before iterating through all types of data and model types to measure time taken to find best model
        train_time_start = time.clock()

        # Iterate through all input types (bag-of-words and TFIDF)
        for input_type_name, (train, valid, test) in input_types.items():

            # Create a results dict for each input type
            results[input_type_name] = {}

            # Cycle through each type of model
            for model_name, model_class in self.model_dict.items():

                # Initialise the model, and train it on the training data
                model = model_class()
                model.fit(train, train_df.baseline_label)
                valid_preds = model.predict(valid)
                test_preds = model.predict(test)

                per_model_results = {}
                for metric_name, metric_fn in self.metrics.items():
                    per_model_results[metric_name] = {}

                    # If we are doing multiclass classification, then we want the f1 score for all classes.
                    # If we are doing binary, then we do not want the 0 and the 1 f1 score, just the 1 f1 score.
                    if metric_name in ["f1", "precision", "recall"] and not multiclass:
                        applied_metric_fn = partial(metric_fn, average="binary")
                    else:
                        applied_metric_fn = partial(metric_fn, average=None)

                    if multiclass:
                        valid_score = applied_metric_fn(binarizer.transform(valid_df.baseline_label), binarizer.transform(valid_preds))
                        test_score = applied_metric_fn(binarizer.transform(test_df.baseline_label), binarizer.transform(test_preds))
                    else:
                        valid_score = applied_metric_fn(valid_df.baseline_label, valid_preds)
                        test_score = applied_metric_fn(test_df.baseline_label, test_preds)

                    per_model_results[metric_name]["valid"] = valid_score
                    per_model_results[metric_name]["test"] = test_score

                    if metric_name == best_metric:
                        if best_score is None or valid_score > best_score:
                            best_score = valid_score
                            best_model = model
                            best_config = (input_type_name, model_name)

                results[input_type_name][model_name] = per_model_results

        train_time_end = time.clock()

        best_results.update({
            "best score": best_score,
            "best model": best_model,
            "best config": best_config,
            "time taken to achieve": train_time_end - train_time_start
        })

        return best_results, results


    def get_baselines(self, input_train_df, input_valid_df, input_test_df, best_metric, is_multilabel):

        per_label_results = {}
        best_per_label_results = {}

        if is_multilabel:
            for column in [x for x in input_train_df.columns if "label_" in x]:
                input_train_df["baseline_label"] = input_train_df[column]
                input_valid_df["baseline_label"] = input_valid_df[column]
                input_test_df["baseline_label"] = input_test_df[column]

                best_per_label_results[column], per_label_results[column] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric)
        else:
            input_train_df["baseline_label"] = input_train_df["label"]
            input_valid_df["baseline_label"] = input_valid_df["label"]
            input_test_df["baseline_label"] = input_test_df["label"]
            best_per_label_results["multiclass"], per_label_results["multiclass"] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric, multiclass=True)

            # Only do binary models if the labels are not already binary
            if len(input_test_df["label"].unique()) > 2:
                for label in input_train_df.label.unique():
                    input_train_df["baseline_label"] = input_train_df["label"] == label
                    input_valid_df["baseline_label"] = input_valid_df["label"] == label
                    input_test_df["baseline_label"] = input_test_df["label"] == label

                    best_per_label_results[label], per_label_results[label] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric)

        return best_per_label_results, per_label_results
