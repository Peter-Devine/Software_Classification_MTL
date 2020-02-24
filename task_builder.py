import requests, zipfile, io, os, json
import pandas as pd
from task import Task
from models import get_language_model

class TaskBuilder:
    def __init__(self, random_state):
        self.random_state = random_state
        self.task_dict = {
            "maalej_2015": Task(data_getter_fn=get_maalej_2015, is_multilabel=False, random_state=random_state)
        }

    def build_tasks(self, names_of_datasets, PARAMS):
        target_task_dict = {}
        language_model = get_language_model(PARAMS.lm_model_name)
        for dataset_name in names_of_datasets:
            target_task_dict[dataset_name] = self.task_dict[dataset_name].build_task(language_model, PARAMS)
        return target_task_dict


    ######### INDIVIDUAL DATA GETTERS ############

    def get_maalej_2015(self):

        if not os.path.exists("REJ_data.zip"):
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()

        # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
        # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
        json_path = "./REJ_data/all.json"

        with open(json_path) as json_file:
            data = json.load(json_file)

        df = pd.DataFrame({"text": [x["title"] + " [SEP] " + x["comment"] if x["title"] is not None else " [SEP] " + x["comment"] for x in data], "label":[x["label"] for x in data]})

        train_val_idx = df.sample(frac=0.7, random_state=self.random_state).index
        test_idx = df.drop(train_val_idx).index
        train_idx = df[train_val_idx].sample(frac=0.85, random_state=self.random_state).index
        valid_idx = df[train_val_idx].drop(train_idx).index

        return df[train_idx], df[valid_idx], df[test_idx]
