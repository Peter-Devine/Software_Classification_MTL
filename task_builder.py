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
            "chen_2014_swiftkey": Task(data_getter_fn=self.get_chen_2014_swiftkey, is_multilabel=False)
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
        # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
        # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
        if not os.path.exists(os.path.join(self.data_path, "REJ_data.zip")):
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = self.data_path)

        json_path = os.path.join(self.data_path, "REJ_data", "all.json")

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
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
        if not os.path.exists(os.path.join(self.data_path, "datasets.zip")):
            r = requests.get("https://sites.google.com/site/appsuserreviews/home/datasets.zip?attredirects=0&d=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=self.data_path)

        def df_getter(data_path, label):
            with open(data_path, "r") as f:
                data = f.read()
            return pd.DataFrame({"text": [" ".join(x.split()[2:]) for x in data.split("\n") if len(x) > 0],
                          "label": label})

        train_info = df_getter(os.path.join(self.data_path, "datasets", "swiftkey", "trainL", "info.txt"), "informative")
        train_noninfo = df_getter(os.path.join(self.data_path, "datasets", "swiftkey", "trainL", "non-info.txt"), "non-informative")

        test_info = df_getter(os.path.join(self.data_path, "datasets", "swiftkey", "test", "info.txt"), "informative")
        test_noninfo = df_getter(os.path.join(self.data_path, "datasets", "swiftkey", "test", "non-info.txt"), "non-informative")

        train_and_val = train_info.append(train_noninfo)

        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = test_info.append(test_noninfo)

        return train, val, test
