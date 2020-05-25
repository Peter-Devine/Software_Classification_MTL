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
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score

from xgboost import XGBClassifier

import numpy as np

import pandas as pd
from functools import partial
import time

class BaselineModels:
    def __init__(self, random_state):
        self.model_dict = {
            "Gaussian NB var_smoothing 1e-11": partial(GaussianNB, var_smoothing=1e-11),
            "Gaussian NB var_smoothing 1e-10": partial(GaussianNB, var_smoothing=1e-10),
            "Gaussian NB var_smoothing 1e-9": partial(GaussianNB, var_smoothing=1e-9),
            "Gaussian NB var_smoothing 1e-8": partial(GaussianNB, var_smoothing=1e-8),
            "Gaussian NB var_smoothing 1e-7": partial(GaussianNB, var_smoothing=1e-7),
            "Complement NB alpha 0": partial(ComplementNB, alpha=1e-10),
            "Complement NB alpha 0.5": partial(ComplementNB, alpha=0.5),
            "Complement NB alpha 1": partial(ComplementNB, alpha=1),
            "Complement NB alpha 2": partial(ComplementNB, alpha=2),
            "Complement NB alpha 5": partial(ComplementNB, alpha=10),
            "Bernoulli NB alpha 0": partial(BernoulliNB, alpha=1e-10),
            "Bernoulli NB alpha 0.5": partial(BernoulliNB, alpha=0.5),
            "Bernoulli NB alpha 1": partial(BernoulliNB, alpha=1),
            "Bernoulli NB alpha 2": partial(BernoulliNB, alpha=2),
            "Bernoulli NB alpha 5": partial(BernoulliNB, alpha=5),
            "Multinomial NB alpha 0": partial(MultinomialNB, alpha=1e-10),
            "Multinomial NB alpha 0.5": partial(MultinomialNB, alpha=0.5),
            "Multinomial NB alpha 1": partial(MultinomialNB, alpha=1),
            "Multinomial NB alpha 2": partial(MultinomialNB, alpha=2),
            "Multinomial NB alpha 5": partial(MultinomialNB, alpha=5),
            "Decision tree classifier max_depth None criterion gini": partial(DecisionTreeClassifier, random_state=random_state, max_depth=None, criterion="gini"),
            "Decision tree classifier max_depth 1 criterion gini": partial(DecisionTreeClassifier, random_state=random_state, max_depth=1, criterion="gini"),
            "Decision tree classifier max_depth 5 criterion gini": partial(DecisionTreeClassifier, random_state=random_state, max_depth=5, criterion="gini"),
            "Decision tree classifier max_depth 10 criterion gini": partial(DecisionTreeClassifier, random_state=random_state, max_depth=10, criterion="gini"),
            "Decision tree classifier max_depth 20 criterion gini": partial(DecisionTreeClassifier, random_state=random_state, max_depth=20, criterion="gini"),
            "Decision tree classifier max_depth None criterion entropy": partial(DecisionTreeClassifier, random_state=random_state, max_depth=None, criterion="entropy"),
            "Decision tree classifier max_depth 1 criterion entropy": partial(DecisionTreeClassifier, random_state=random_state, max_depth=1, criterion="entropy"),
            "Decision tree classifier max_depth 5 criterion entropy": partial(DecisionTreeClassifier, random_state=random_state, max_depth=5, criterion="entropy"),
            "Decision tree classifier max_depth 10 criterion entropy": partial(DecisionTreeClassifier, random_state=random_state, max_depth=10, criterion="entropy"),
            "Decision tree classifier max_depth 20 criterion entropy": partial(DecisionTreeClassifier, random_state=random_state, max_depth=20, criterion="entropy"),
            "Support vector classifier C 0.1": partial(SVC, random_state=random_state, C=0.1, probability=True),
            "Support vector classifier C 0.5": partial(SVC, random_state=random_state, C=0.5, probability=True),
            "Support vector classifier C 1": partial(SVC, random_state=random_state, C=1, probability=True),
            "Support vector classifier C 2": partial(SVC, random_state=random_state, C=2, probability=True),
            "Support vector classifier C 5": partial(SVC, random_state=random_state, C=5, probability=True),
            "Gradient boosting classifier max_depth 1 subsample 1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=1, subsample=1),
            "Gradient boosting classifier max_depth 1 subsample 0.5": partial(GradientBoostingClassifier, random_state=random_state, max_depth=1, subsample=0.5),
            "Gradient boosting classifier max_depth 1 subsample 0.1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=1, subsample=0.1),
            "Gradient boosting classifier max_depth 3 subsample 1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=3, subsample=1),
            "Gradient boosting classifier max_depth 3 subsample 0.5": partial(GradientBoostingClassifier, random_state=random_state, max_depth=3, subsample=0.5),
            "Gradient boosting classifier max_depth 3 subsample 0.1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=3, subsample=0.1),
            "Gradient boosting classifier max_depth 5 subsample 1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=5, subsample=1),
            "Gradient boosting classifier max_depth 5 subsample 0.5": partial(GradientBoostingClassifier, random_state=random_state, max_depth=5, subsample=0.5),
            "Gradient boosting classifier max_depth 5 subsample 0.1": partial(GradientBoostingClassifier, random_state=random_state, max_depth=5, subsample=0.1),
            "XG boosting classifier max_depth 1 subsample 1": partial(XGBClassifier, random_state=random_state, max_depth=1, subsample=1),
            "XG boosting classifier max_depth 1 subsample 0.5": partial(XGBClassifier, random_state=random_state, max_depth=1, subsample=0.5),
            "XG boosting classifier max_depth 1 subsample 0.1": partial(XGBClassifier, random_state=random_state, max_depth=1, subsample=0.1),
            "XG boosting classifier max_depth 3 subsample 1": partial(XGBClassifier, random_state=random_state, max_depth=3, subsample=1),
            "XG boosting classifier max_depth 3 subsample 0.5": partial(XGBClassifier, random_state=random_state, max_depth=3, subsample=0.5),
            "XG boosting classifier max_depth 3 subsample 0.1": partial(XGBClassifier, random_state=random_state, max_depth=3, subsample=0.1),
            "XG boosting classifier max_depth 6 subsample 1": partial(XGBClassifier, random_state=random_state, max_depth=6, subsample=1),
            "XG boosting classifier max_depth 6 subsample 0.5": partial(XGBClassifier, random_state=random_state, max_depth=6, subsample=0.5),
            "XG boosting classifier max_depth 6 subsample 0.1": partial(XGBClassifier, random_state=random_state, max_depth=6, subsample=0.1)
        }

        self.metrics = {
            "accuracy": accuracy_score,
            "f1": partial(fbeta_score, beta=1, average=None),
            "f2": partial(fbeta_score, beta=2, average=None),
            "precision": partial(precision_score, average=None),
            "recall": partial(recall_score, average=None),
            "average f1": partial(fbeta_score, beta=1, average="macro"),
            "average f2": partial(fbeta_score, beta=2, average="macro"),
            "average precision": partial(precision_score, average="macro"),
            "average recall": partial(recall_score, average="macro"),
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
            test_mtl_df = self.create_zero_shot_df(test_task.test_df, zero_shot_label, test_task.is_multilabel, training=False)

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
            # We keep the labels the same for this baseline. In other words, we train a multiclass classifier for this zero-shot learning test, and convert the outputs later.
            task.train_df["baseline_label"] = task.train_df.label
            task.valid_df["baseline_label"] = task.valid_df.label
            best_results, results = self.get_baseline_results(best_metric=best_metric, train_df=task.train_df, valid_df=task.valid_df, test_df=None, is_multiclass=False)

            for test_task_name, test_task in test_task_dict.items():
                zero_shot_results[task_name][test_task_name] = self.get_zero_shot_result_given_model(task.train_df, task.valid_df, test_task.test_df, best_results, zero_shot_label=zero_shot_label)

        return zero_shot_results

    def get_zero_shot_result_given_model(self, train_df, valid_df, test_df, best_results, zero_shot_label=None):
        is_best_model_tfidf = best_results["best config"]["input type"] == "tfidf"
        train_X, valid_X, test_X = self.transform_text(train_df, valid_df, test_df, is_idf=is_best_model_tfidf)

        best_model = best_results["best model"]

        # If we have a zero-shot label (I.e., the datasets have not already been converted to a binary label set of True/False),
        # then we need to get the probabilities of the zero-shot label class, and then compare that probability to that of every other class.
        # Then, if the probability of that class is greater than the sum of the probabilities of every other class, then we predict that class.
        # If not, then we predict another class.
        if zero_shot_label is not None:
            zero_shot_class_index = [i for i, class_ in enumerate(best_model.classes_) if zero_shot_label in class_.lower()]
            assert len(zero_shot_class_index) == 1, f"Multiple classes contain zero shot label. Looking for one instance of {zero_shot_label} within {best_model.classes_}"
            zero_shot_class_index = zero_shot_class_index[0]

            # Get the probabilities that the prediction is the in class zero shot label
            test_preds = best_model.predict_proba(test_X)[:,zero_shot_class_index]
            # We set the prediction that the label is the zero-shot class if its probability is greater than 0.5 (50%).
            # Since this is a binary problem (zero-shot class or NOT zero-shot class), a threshold of 50% probability is appropriate to choose 1 class between 2.
            test_preds = np.array([pred >= 0.5 for pred in test_preds])
            test_golds = np.array([zero_shot_label in gold.lower() for gold in test_df.label])
        else:
            test_preds = best_model.predict(test_X)
            test_golds = test_df.baseline_label

        zero_shot_result = self.get_metrics_from_preds(test_golds, test_preds, is_multiclass=False)
        return zero_shot_result

    def get_metrics_from_preds(self, golds, preds, is_multiclass, binarizer=None):
        metrics_results = {}

        # Cycle through all metrics, getting the metric given golds, preds for each and saving said result
        for metric_name, metric_fn in self.metrics.items():

            if is_multiclass:
                assert binarizer is not None, "Multiclass metrics requested without binarizer passed."
                score = metric_fn(binarizer.transform(golds), binarizer.transform(preds))
            else:
                score = metric_fn(golds, preds)

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
                    all_best_metrics = {"valid results best": valid_results}
                    if test is not None:
                        all_best_metrics["test results best"] = test_results

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

                # Just optimize over binary f1 instead of average for binary classification
                binary_best_metric = best_metric.replace("average ", "")

                best_per_label_results[column], per_label_results[column] = self.get_baseline_results(best_metric=binary_best_metric, train_df=input_train_df, valid_df=input_valid_df, test_df=input_test_df, is_multiclass=False)
        else:
            # Make multiclass label set by simply copying labels to our temporary baseline_label column
            input_train_df["baseline_label"] = input_train_df["label"]
            input_valid_df["baseline_label"] = input_valid_df["label"]
            input_test_df["baseline_label"] = input_test_df["label"]
            best_per_label_results["multiclass"], per_label_results["multiclass"] = self.get_baseline_results(best_metric=best_metric, train_df=input_train_df, valid_df=input_valid_df, test_df=input_test_df, is_multiclass=True)

        return best_per_label_results, per_label_results
