from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class BaselineModels:
    def __init__(self, is_multilabel):
        self.is_multilabel = is_multilabel
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
            "precision": precision_score,
            "recall": recall_score
        }

    # TODO: Get Zero shot baslines
    # def get_zero_shot_baselines(self, list_of_datasets, best_metric="f1"):
    #     for input_train_df, input_valid_df, input_test_df in list_of_datasets:
    #         best_per_label_results, per_label_results = self.get_baselines(input_train_df, input_valid_df, input_test_df, best_metric)

    def get_results_on_binary_task(self, train_df, valid_df, test_df, best_metric):

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

        best_score = None
        best_model = None
        best_config = None

        for input_type_name, (train, valid, test) in input_types.items():
            results[input_type_name] = {}
            for model_name, model_class in self.model_dict.items():
                model = model_class()
                model.fit(train, train_df.baseline_label)
                valid_preds = model.predict(valid)
                test_preds = model.predict(test)

                if model_name == "Gradient boosting regressor":
                    valid_preds = valid_preds > 0.5
                    test_preds = test_preds > 0.5

                per_model_results = {}
                for metric_name, metric_fn in self.metrics.items():
                    per_model_results[metric_name] = {}

                    valid_score = metric_fn(valid_df.baseline_label, valid_preds)
                    test_score = metric_fn(test_df.baseline_label, test_preds)

                    per_model_results[metric_name]["valid"] = valid_score
                    per_model_results[metric_name]["test"] = test_score

                    if metric_name == best_metric:
                        if best_score is None or valid_score > best_score:
                            best_score = valid_score
                            best_model = model
                            best_config = (input_type_name, model_name)

                results[input_type_name][model_name] = per_model_results

        best_results = {
            "best score": best_score,
            "best model": best_model,
            "best config": best_config
        }

        return best_results, results


    def get_baselines(self, input_train_df, input_valid_df, input_test_df, best_metric):

        per_label_results = {}
        best_per_label_results = {}

        if self.is_multilabel:
            for column in [x for x in input_train_df.columns if "label_" in x]:
                input_train_df["baseline_label"] = input_train_df[column]
                input_valid_df["baseline_label"] = input_valid_df[column]
                input_test_df["baseline_label"] = input_test_df[column]

                best_per_label_results[column], per_label_results[column] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric)
        else:
            input_train_df["baseline_label"] = input_train_df["label"]
            input_valid_df["baseline_label"] = input_valid_df["label"]
            input_test_df["baseline_label"] = input_test_df["label"]
            best_per_label_results["multiclass"], per_label_results["multiclass"] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric)
            for label in input_train_df.label.unique():
                input_train_df["baseline_label"] = input_train_df["label"] == label
                input_valid_df["baseline_label"] = input_valid_df["label"] == label
                input_test_df["baseline_label"] = input_test_df["label"] == label

                best_per_label_results[label], per_label_results[label] = self.get_results_on_binary_task(input_train_df, input_valid_df, input_test_df, best_metric)

        return best_per_label_results, per_label_results