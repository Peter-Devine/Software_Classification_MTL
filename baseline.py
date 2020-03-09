from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
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
            "Support vector classifier": SVC,
            "Gradient boosting regressor": GradientBoostingRegressor
        }
        self.metrics = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score
        }

    def get_baselines(self, input_train_df, input_valid_df, input_test_df):

        def get_results_on_binary_task(train_df, valid_df, test_df):

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

            X_train_bow, X_valid_bow, X_test_bow = transform_text(train_df, valid_df, test_df, is_idf=False)
            X_train_tfidf, X_valid_tfidf, X_test_tfidf = transform_text(train_df, valid_df, test_df, is_idf=True)

            def train_and_eval_on_dataset(train, valid, test):
                per_model_results = {}
                for model_name, model_class in self.model_dict.items():
                    model = model_class()
                    model.fit(train, train_df.baseline_label)
                    valid_preds = model.predict(valid)
                    test_preds = model.predict(test)

                    if model_name == "Gradient boosting regressor":
                        valid_preds = valid_preds > 0.5
                        test_preds = test_preds > 0.5

                    results = {}
                    for metric_name, metric_fn in self.metrics.items():
                        results[metric_name] = {}
                        results[metric_name]["valid"] = metric_fn(valid_df.baseline_label, valid_preds)
                        results[metric_name]["test"] = metric_fn(test_df.baseline_label, test_preds)

                    per_model_results[model_name] = results

                return per_model_results

            bow_results = train_and_eval_on_dataset(X_train_bow, X_valid_bow, X_test_bow)
            tfidf_results = train_and_eval_on_dataset(X_train_tfidf, X_valid_tfidf, X_test_tfidf)

            return {
                "bow": bow_results,
                "tfidf": tfidf_results
            }

        per_label_results = {}

        if self.is_multilabel:
            for column in [x for x in input_train_df.columns if "label_" in x]:
                input_train_df["baseline_label"] = input_train_df[column]
                input_valid_df["baseline_label"] = input_valid_df[column]
                input_test_df["baseline_label"] = input_test_df[column]

                per_label_results[column] = get_results_on_binary_task(input_train_df, input_valid_df, input_test_df)
        else:
            for label in input_train_df.label.unique():
                input_train_df["baseline_label"] = input_train_df["label"] == label
                input_valid_df["baseline_label"] = input_valid_df["label"] == label
                input_test_df["baseline_label"] = input_test_df["label"] == label

                per_label_results[label] = get_results_on_binary_task(input_train_df, input_valid_df, input_test_df)

        best_per_label_results = {}

        for label, label_results in per_label_results.items():
            top_performance_dict = {}
            for feat_name, feat_results in label_results.items():
                for model_name, model_results in feat_results.items():
                    for metric_name, metric_results in model_results.items():
                        if metric_name not in top_performance_dict.keys() or metric_results["valid"] > top_performance_dict[metric_name]["valid"]:
                            top_performance_dict[metric_name] = {
                                "feat": feat_name,
                                "model": model_name,
                                "valid": metric_results["valid"],
                                "test": metric_results["test"]
                            }
            best_per_label_results[label] = top_performance_dict

        return best_per_label_results, per_label_results