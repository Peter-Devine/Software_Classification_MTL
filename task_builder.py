import requests, zipfile, io, os, json
import pandas as pd
from task import Task
from models import get_language_model

class TaskBuilder:
    def __init__(self, random_state):
        self.random_state = random_state
        self.task_dict = {
            "maalej_2015": Task(data_getter_fn=self.get_maalej_2015, is_multilabel=False),
            "maalej_2015_bug_bin": Task(data_getter_fn=self.get_maalej_2015_bug_bin, is_multilabel=False),
            "chen_2014_swiftkey": Task(data_getter_fn=self.get_chen_2014_swiftkey, is_multilabel=False),
            "ciurumelea_2017_fine": Task(data_getter_fn=self.get_ciurumelea_2017_fine, is_multilabel=True),
            "ciurumelea_2017_coarse": Task(data_getter_fn=self.get_ciurumelea_2017_coarse, is_multilabel=True)
        }
        self.data_path = "./data"

    def build_tasks(self, names_of_datasets, PARAMS):
        target_task_dict = {}
        language_model = get_language_model(PARAMS.lm_model_name)
        for dataset_name in names_of_datasets:
            target_task_dict[dataset_name] = self.task_dict[dataset_name].build_task(language_model, PARAMS)
        return target_task_dict

    ######### INDIVIDUAL DATA GETTERS ############

    def get_maalej_2015(self):
        task_data_path = os.path.join(self.data_path, "maalej_2015")
        # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
        # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
        if not os.path.exists(os.path.join(task_data_path, "REJ_data.zip")):
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = task_data_path)

        json_path = os.path.join(task_data_path, "REJ_data", "all.json")

        with open(json_path) as json_file:
            data = json.load(json_file)

        df = pd.DataFrame({"text": [
            "Title: " + x["title"] + " Comment: " + x["comment"] if x["title"] is not None else "Comment: " + x[
                "comment"] for x in data], "label": [x["label"] for x in data]}).sample(n=100,
                                                                                        random_state=self.random_state)

        train_val_idx = df.sample(frac=0.7, random_state=self.random_state).index
        test_idx = df.drop(train_val_idx).index
        train_idx = df.loc[train_val_idx].sample(frac=0.85, random_state=self.random_state).index
        valid_idx = df.loc[train_val_idx].drop(train_idx).index

        return df.loc[train_idx], df.loc[valid_idx], df.loc[test_idx]


    def get_maalej_2015_bug_bin(self):
        train, dev, test = self.get_maalej_2015()

        train["label"] = train.label.apply(lambda x: x if x=="Bug" else "NoBug")
        dev["label"] = dev.label.apply(lambda x: x if x=="Bug" else "NoBug")
        test["label"] = test.label.apply(lambda x: x if x=="Bug" else "NoBug")

        return train, dev, test

    def get_chen_2014_swiftkey(self):
        task_data_path = os.path.join(self.data_path, "chen_2014")
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
        if not os.path.exists(os.path.join(task_data_path, "datasets.zip")):
            r = requests.get("https://sites.google.com/site/appsuserreviews/home/datasets.zip?attredirects=0&d=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=task_data_path)

        def df_getter(data_path, label):
            with open(data_path, "r") as f:
                data = f.read()
            return pd.DataFrame({"text": [" ".join(x.split()[2:]) for x in data.split("\n") if len(x) > 0],
                          "label": label})

        train_info = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "trainL", "info.txt"), "informative")
        train_noninfo = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "trainL", "non-info.txt"), "non-informative")

        test_info = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "test", "info.txt"), "informative")
        test_noninfo = df_getter(os.path.join(task_data_path, "datasets", "swiftkey", "test", "non-info.txt"), "non-informative")

        train_and_val = train_info.append(train_noninfo)

        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = test_info.append(test_noninfo)

        return train, val, test

    def get_ciurumelea_2017_fine(self):
        train, val, test = self.get_ciurumelea_2017()

        def remove_coarse_labels(df):
            coarse_columns = ["COMPATIBILITY", "USAGE", "PRICING", "PROTECTION", "RESSOURCES", "OTHER"]
            return df.drop([f"label_{x}" for x in coarse_columns], axis=1)

        train = remove_coarse_labels(train)
        val = remove_coarse_labels(val)
        test = remove_coarse_labels(test)

        return train, val, test

    def get_ciurumelea_2017_coarse(self):
        train, val, test = self.get_ciurumelea_2017()

        def combine_fine_labels_to_coarse(df):
            coarse_to_fine_dict = {
                "COMPATIBILITY": ['DEVICE', 'ANDROID VERSION', 'HARDWARE'],
                "USAGE": ['APP USABILITY', 'UI'],
                "RESSOURCES": ['PERFORMANCE', 'BATTERY', 'MEMORY'],
                "PRICING": ['LICENSING', 'PRICE'],
                "PROTECTION": ['SECURITY', 'PRIVACY'],
                "OTHER": ['ERROR']
            }

            for coarse_column, fine_mappings in coarse_to_fine_dict.items():
                fine_mappings_column_names = [f"label_{x}" for x in fine_mappings]
                df[f"label_{coarse_column}"] = df[f"label_{coarse_column}"] | df[fine_mappings_column_names].any(axis=1)
                df = df.drop(fine_mappings_column_names, axis=1)

            return df

        train = combine_fine_labels_to_coarse(train)
        val = combine_fine_labels_to_coarse(val)
        test = combine_fine_labels_to_coarse(test)

        return train, val, test

    def get_ciurumelea_2017(self):
        task_data_path = os.path.join(self.data_path, "ciurumelea_2017")
        # from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7884612&tag=1
        # Analyzing Reviews and Code of Mobile Apps for Better Release Planning
        if not os.path.exists(os.path.join(task_data_path, "UserReviewReference-Replication-Package-URR-v1.0.zip")):
            r = requests.get("https://zenodo.org/record/161842/files/panichella/UserReviewReference-Replication-Package-URR-v1.0.zip?download=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=task_data_path)

        review_data_path = os.path.join(task_data_path, "panichella-UserReviewReference-Replication-Package-643afe0", "data", "reviews", "golden_set.csv")

        df = pd.read_csv(review_data_path, encoding="iso-8859-1")
        df["text"] = df.reviewText

        label_columns = ["classification", "class1", "class2", "class3", "class4", "class5"]

        unique_values = set()
        for column in label_columns:
            unique_values.update(df[column].unique())

        print(unique_values)

        for unique_value in unique_values:
            if type(unique_value) == str and unique_value != "COMPATIBILTY":
                df[f"label_{unique_value}"] = (df[label_columns] == unique_value).any(axis=1)

        # There is a spellling mistake in the dataframe where COMPATIBILITY is mistakenly labelled COMPATIBILTY, but only for some reviews. We fix this here.
        # We choose to ignore the spelling mistake of labelling RESOURCES as RESSOURCES, and just continue with that typo throughout the experiment.
        df[f"label_COMPATIBILITY"] = df[f"label_COMPATIBILITY"] | (df[label_columns] == "COMPATIBILTY").any(axis=1)

        columns_to_include = ["text"] + [x for x in df.columns if "label_" in x]
        df = df[columns_to_include]

        train_and_val = df.sample(frac=0.8, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test