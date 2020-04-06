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
import pandas as pd
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

    # Gets the zero-shot ability of classical models trained on all datasets save one
    # If is_zero_shot, we train the model on all tasks's combined train/validation set (except one) and test on another task's test set
    # If not is_zero_shot, then we train the model on all the tasks combined datasets, and then we test on each task's test set
    def get_MTL_baselines(self, train_task_dict, test_task_dict, best_metric, zero_shot_label):
        zero_shot_mtl_results = {}

        # Combine all training datasets into one big binary dataset
        train_mtl_df, valid_mtl_df = self.combine_task_datasets(train_task_dict, zero_shot_label)

        best_results, results = self.get_baseline_results(best_metric=best_metric, train_df=train_mtl_df, valid_df=valid_mtl_df, test_df=None, is_multiclass=False)

        # Iterate over all tasks as our test tasks
        for test_task_name, test_task in test_task_dict.items():

            # Get the test set
            test_mtl_df = self.create_zero_shot_df(test_task.train_df, zero_shot_label, test_task.is_multilabel, training=False)

            # Get results on test set using model trained on training set and selected on validation set
            zero_shot_mtl_results[test_task_name] = self.get_zero_shot_result_given_model(train_mtl_df, valid_mtl_df, test_mtl_df, best_results)

        return zero_shot_mtl_results

    def combine_task_datasets(self, task_dict, zero_shot_label):

        train_mtl_df = None
        valid_mtl_df = None

        # Append all training and validation sets together
        for train_task_name, train_task in task_dict.items():
            train_zero_shot_df = self.create_zero_shot_df(train_task.train_df, zero_shot_label, train_task.is_multilabel, training=True)
            valid_zero_shot_df = self.create_zero_shot_df(train_task.valid_df, zero_shot_label, train_task.is_multilabel, training=False)

            if train_mtl_df is None:
                train_mtl_df = train_zero_shot_df
            else:
                train_mtl_df = train_mtl_df.append(train_zero_shot_df)

            if valid_mtl_df is None:
                valid_mtl_df = valid_zero_shot_df
            else:
                valid_mtl_df = valid_mtl_df.append(valid_zero_shot_df)

        return train_mtl_df, valid_mtl_df

    def create_zero_shot_df(self, df, zero_shot_label, is_multilabel, training=False):
        if is_multilabel:
            # If multilabel, get the column which has our target label in the title (And assert that it indeed is there)
            label_column_names = [x for x in df.columns if zero_shot_label in x.lower()]
            assert len(label_column_names) == 1, f"Not exactly one column name that corresponds to zero shot label {zero_shot_label} ({df.columns})"
            label_column_names = label_column_names[0]
            df_labels = df[label_column_name].apply(lambda x: True if x else False)
        else:
            # Only check if there are any positive labels in train set (I.e. not valid/test set) as without positive labels we cannot train
            if training:
                assert any([zero_shot_label in x.lower() for x in df.label.unique()]), f"Zero shot label, {zero_shot_label}, not in dataset (Vals: {df.label.unique()})"
            df_labels = df.label.apply(lambda x: True if zero_shot_label in x.lower() else False)

        zero_shot_df = pd.DataFrame({"text": df.text, "baseline_label": df_labels})
        return zero_shot_df

    # Gets the zero-shot ability of classical models trained on a given dataset
    # We train the model on one task's train/validation set and test on another task's test set
    # We then map both dataset's label sets to a given label (E.g. Bug/ No bug) as a list of booleans
    def get_zero_shot_baselines(self, train_task_dict, test_task_dict, best_metric, zero_shot_label):
        zero_shot_results = {}
        for task_name, task in train_task_dict.items():
            train_df = self.create_zero_shot_df(task.train_df, zero_shot_label, task.is_multilabel, training=True)
            valid_df = self.create_zero_shot_df(task.valid_df, zero_shot_label, task.is_multilabel, training=False)
            best_results, results = self.get_baseline_results(best_metric=best_metric, train_df=train_df, valid_df=valid_df, test_df=None, is_multiclass=False)

            # Store the pre-evaluation model config, metrics etc. in the results
            zero_shot_results[task_name] = best_results

            for test_task_name, test_task in test_task_dict.items():
                test_df = self.create_zero_shot_df(test_task.test_df, zero_shot_label, test_task.is_multilabel, training=False)

                zero_shot_results[task_name][test_task_name] = self.get_zero_shot_result_given_model(train_df, valid_df, test_df, best_results)

        return zero_shot_results

    def get_zero_shot_result_given_model(self, train_df, valid_df, test_df, best_results):
        is_best_model_tfidf = best_results["best config"]["input type"] == "tfidf"
        train_X, valid_X, test_X = self.transform_text(train_df, valid_df, test_df, is_idf=is_best_model_tfidf)

        best_model = best_results["best model"]
        test_preds = best_model.predict(test_X)

        zero_shot_result = self.get_metrics_from_preds(test_df.baseline_label, test_preds, is_multiclass=False)
        return zero_shot_result

    def get_metrics_from_preds(self, golds, preds, is_multiclass, binarizer=None):
        metrics_results = {}
        for metric_name, metric_fn in self.metrics.items():

            # If we are doing multiclass classification, then we want the f1 score for all classes.
            # If we are doing binary, then we do not want the 0 and the 1 f1 score, just the 1 f1 score.
            if metric_name in ["f1", "precision", "recall"]:
                if not is_multiclass:
                    applied_metric_fn = partial(metric_fn, average="binary")
                else:
                    applied_metric_fn = partial(metric_fn, average=None)
            else:
                applied_metric_fn = metric_fn

            if is_multiclass:
                assert binarizer is not None, "Multiclass metrics requested without binarizer passed."
                score = applied_metric_fn(binarizer.transform(golds), binarizer.transform(preds))
            else:
                score = applied_metric_fn(golds, preds)

            metrics_results[metric_name] = score

        return metrics_results

    def transform_text(self, train, valid, test, is_idf):
        bow_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfTransformer()

        # Get BOW features for each split
        train_feat = bow_vectorizer.fit_transform(train.text).toarray()
        valid_feat = bow_vectorizer.transform(valid.text).toarray()
        if test is not None:
            test_feat = bow_vectorizer.transform(test.text).toarray()
        else:
            test_feat = None

        if is_idf:
            # Convert BOW features into TFIDF features
            train_feat = tfidf_vectorizer.fit_transform(train_feat).toarray()
            valid_feat = tfidf_vectorizer.transform(valid_feat).toarray()
            if test is not None:
                test_feat = tfidf_vectorizer.transform(test_feat).toarray()

        return train_feat, valid_feat, test_feat

    def get_baseline_results(self, best_metric, train_df, valid_df, test_df=None, is_multiclass=False):

        bow_splits = self.transform_text(train_df, valid_df, test_df, is_idf=False)
        tfidf_splits = self.transform_text(train_df, valid_df, test_df, is_idf=True)

        input_types = {
            "bow": bow_splits,
            "tfidf": tfidf_splits
        }

        results = {}

        # Save the best metric, model and configuration (model name and input type) based on the validation score
        best_score = None
        all_best_metrics = None
        best_model = None
        best_config = None
        best_results = {}

        # If we are doing a multiclass baseline, then we need to know which labels are referred to by which columns in our prec, rec, f1 output
        # We fit an sklearn binarizer on the training data and output the mappings of these classes to our results output
        if is_multiclass:
            binarizer = LabelBinarizer()
            binarizer.fit(train_df.baseline_label)
            results["multiclass label map"] = binarizer.classes_
            best_results["multiclass label map"] = binarizer.classes_
        else:
            binarizer = None

        # Start a clock before iterating through all types of data and model types to measure time taken to find best model
        train_time_start = time.time()

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
                if test is not None:
                    test_preds = model.predict(test)

                per_model_results = {}
                valid_results = self.get_metrics_from_preds(valid_df.baseline_label, valid_preds, is_multiclass, binarizer=binarizer)
                per_model_results["valid"] = valid_results
                if test is not None:
                    test_results = self.get_metrics_from_preds(test_df.baseline_label, test_preds, is_multiclass, binarizer=binarizer)
                    per_model_results["test"] = test_results

                # Find the score used to select the best model based on the best_metric
                valid_score = valid_results[best_metric]

                if best_score is None or valid_score > best_score:

                    # Save the score of the best model
                    best_score = valid_score

                    # Save all valid and test score metrics associated with the best model if available
                    all_best_metrics = {"valid results best: ": valid_results}
                    if test is not None:
                        all_best_metrics["test results best: "] = test_results

                    # Save the best model for future use
                    best_model = model
                    best_config = {"input type": input_type_name, "model name": model_name}

                results[input_type_name][model_name] = per_model_results

        train_time_end = time.time()

        best_results.update({
            "best score": best_score,
            "all best metrics": all_best_metrics,
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

                best_per_label_results[column], per_label_results[column] = self.get_baseline_results(best_metric=best_metric, train_df=input_train_df, valid_df=input_valid_df, test_df=input_test_df, is_multiclass=False)
        else:
            # Make multiclass label set by simply copying labels to our temporary baseline_label column
            input_train_df["baseline_label"] = input_train_df["label"]
            input_valid_df["baseline_label"] = input_valid_df["label"]
            input_test_df["baseline_label"] = input_test_df["label"]
            best_per_label_results["multiclass"], per_label_results["multiclass"] = self.get_baseline_results(best_metric=best_metric, train_df=input_train_df, valid_df=input_valid_df, test_df=input_test_df, is_multiclass=True)

            # Only do binary models if the labels are not already binary
            if len(input_test_df["label"].unique()) > 2:
                for label in input_train_df.label.unique():
                    # Make binary label set by making a boolean list of whether the label set is a given label or not
                    input_train_df["baseline_label"] = input_train_df["label"] == label
                    input_valid_df["baseline_label"] = input_valid_df["label"] == label
                    input_test_df["baseline_label"] = input_test_df["label"] == label

                    best_per_label_results[label], per_label_results[label] = self.get_baseline_results(best_metric=best_metric, train_df=input_train_df, valid_df=input_valid_df, test_df=input_test_df, is_multiclass=False)

        return best_per_label_results, per_label_results
