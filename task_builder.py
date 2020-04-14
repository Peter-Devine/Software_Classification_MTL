import requests, zipfile, io, os, json, shutil
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup
from task import Task
from models import get_language_model

class TaskBuilder:
    def __init__(self, random_state):
        self.random_state = random_state
        self.task_dict = {
            "maalej_2016": Task(data_getter_fn=self.get_maalej_2016, is_multilabel=False),
            # "maalej_2016_bug_bin": Task(data_getter_fn=self.get_maalej_2016_bug_bin, is_multilabel=False),
            # "maalej_2016_rating_bin": Task(data_getter_fn=self.get_maalej_2016_rating_bin, is_multilabel=False),
            # "maalej_2016_feature_bin": Task(data_getter_fn=self.get_maalej_2016_feature_bin, is_multilabel=False),
            # "maalej_2016_user_bin": Task(data_getter_fn=self.get_maalej_2016_user_bin, is_multilabel=False),
            # "maalej_2016_user_bin_multilabel": Task(data_getter_fn=self.get_maalej_2016_user_bin_multilabel, is_multilabel=True),
            "williams_2017": Task(data_getter_fn=self.get_williams_2017, is_multilabel=False),
            "chen_2014_swiftkey": Task(data_getter_fn=self.get_chen_2014_swiftkey, is_multilabel=False),
            "ciurumelea_2017_fine": Task(data_getter_fn=self.get_ciurumelea_2017_fine, is_multilabel=True),
            "ciurumelea_2017_coarse": Task(data_getter_fn=self.get_ciurumelea_2017_coarse, is_multilabel=True),
            "di_sorbo_2017": Task(data_getter_fn=self.get_di_sorbo_2017, is_multilabel=False),
            "guzman_2015": Task(data_getter_fn=self.get_guzman_2015, is_multilabel=False)
        }
        self.data_path = "./data"

    def build_tasks(self, names_of_datasets, PARAMS, is_test_tasks=False):
        target_task_dict = {}

        # If we are only doing inference over these tasks, we do not need a model, or shared language model
        if is_test_tasks:
            language_model = None
        else:
            language_model = get_language_model(PARAMS.lm_model_name)

        for dataset_name in names_of_datasets:
            target_task_dict[dataset_name] = self.task_dict[dataset_name].build_task(language_model, PARAMS, is_test_tasks)

        return target_task_dict

    ######### INDIVIDUAL DATA GETTERS ############

    def get_maalej_2016(self):
        task_data_path = os.path.join(self.data_path, "maalej_2016")
        # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
        # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
        zip_file_path = os.path.join(task_data_path, "REJ_data.zip")
        if not os.path.exists(zip_file_path):
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = task_data_path)

        json_path = os.path.join(task_data_path, "REJ_data", "all.json")

        with open(json_path) as json_file:
            data = json.load(json_file)

        shutil.rmtree(task_data_path)

        df = pd.DataFrame({"text": [
            "Title: " + x["title"] + " Comment: " + x["comment"] if x["title"] is not None else "Comment: " + x[
                "comment"] for x in data], "label": [x["label"] for x in data]})

        train_and_val = df.sample(frac=0.8, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test

    def get_maalej_2016_bin(self, label):
        train, dev, test = self.get_maalej_2016()

        train["label"] = train.label.apply(lambda x: x if x==label else f"No{label}")
        dev["label"] = dev.label.apply(lambda x: x if x==label else f"No{label}")
        test["label"] = test.label.apply(lambda x: x if x==label else f"No{label}")

        return train, dev, test

    def get_maalej_2016_bug_bin(self):
        return self.get_maalej_2016_bin("Bug")

    def get_maalej_2016_rating_bin(self):
        return self.get_maalej_2016_bin("Rating")

    def get_maalej_2016_feature_bin(self):
        return self.get_maalej_2016_bin("Feature")

    def get_maalej_2016_user_bin(self):
        return self.get_maalej_2016_bin("UserExperience")

    def get_maalej_2016_user_bin_multilabel(self):
        train, dev, test = self.get_maalej_2016_bin("UserExperience")
        train.rename(columns={'label': 'label_user'}, inplace=True)
        dev.rename(columns={'label': 'label_user'}, inplace=True)
        test.rename(columns={'label': 'label_user'}, inplace=True)

        train["label_user"] = train["label_user"] == "UserExperience"
        dev["label_user"] = dev["label_user"] == "UserExperience"
        test["label_user"] = test["label_user"] == "UserExperience"
        return train, dev, test

    def get_williams_2017(self):
        task_data_path = os.path.join(".", "williams_2017")
        # from
        # Mining Twitter feeds for software user requirements.
        zip_file_path = os.path.join(task_data_path, "re17.zip")
        if not os.path.exists(zip_file_path):
            r = requests.get("http://seel.cse.lsu.edu/data/re17.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = task_data_path)

        file_path = os.path.join(task_data_path, "RE17", "tweets_full_dataset.dat")

        with open(file_path, "r", encoding='ISO-8859-1') as f:
            data = f.read()

        table = pd.read_table(StringIO("\n".join(data.split("\n")[16:])), names=["text_data"])

        pos = table["text_data"].apply(lambda x: x.split(",")[0])
        neg = table["text_data"].apply(lambda x: x.split(",")[1])
        feedback_class = table["text_data"].apply(lambda x: x.split(",")[2])
        content = table["text_data"].apply(lambda x: ",".join(x.split(",")[3:-10]).strip("\"") if len(x.split(",")) >= 10 else None)
        feedback_ids = table["text_data"].apply(lambda x: x.split(",")[-10])
        n_favorites = table["text_data"].apply(lambda x: x.split(",")[-9])
        n_followers = table["text_data"].apply(lambda x: x.split(",")[-8])
        n_friends = table["text_data"].apply(lambda x: x.split(",")[-7])
        n_statuses = table["text_data"].apply(lambda x: x.split(",")[-6])
        n_listed = table["text_data"].apply(lambda x: x.split(",")[-5])
        verified = table["text_data"].apply(lambda x: x.split(",")[-4])
        timezone = table["text_data"].apply(lambda x: x.split(",")[-3])
        is_reply = table["text_data"].apply(lambda x: x.split(",")[-2])
        date_posted = table["text_data"].apply(lambda x: x.split(",")[-1])

        df = pd.DataFrame({
            "pos": pos,
            "neg": neg,
            "label": feedback_class,
            "text": content,
            "feedback_ids": feedback_ids,
            "n_favorites": n_favorites,
            "n_followers": n_followers,
            "n_friends": n_friends,
            "n_statuses": n_statuses,
            "n_listed": n_listed,
            "verified": verified,
            "timezone": timezone,
            "is_reply": is_reply,
            "date_posted": date_posted
        })

        train_and_val = df.sample(frac=0.8, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test



    def get_chen_2014_swiftkey(self):
        task_data_path = os.path.join(self.data_path, "chen_2014")
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
        zip_file_path = os.path.join(task_data_path, "datasets.zip")
        if not os.path.exists(zip_file_path):
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

        shutil.rmtree(task_data_path)


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
        zip_file_path = os.path.join(task_data_path, "UserReviewReference-Replication-Package-URR-v1.0.zip")
        if not os.path.exists(zip_file_path):
            r = requests.get("https://zenodo.org/record/161842/files/panichella/UserReviewReference-Replication-Package-URR-v1.0.zip?download=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=task_data_path)

        review_data_path = os.path.join(task_data_path, "panichella-UserReviewReference-Replication-Package-643afe0", "data", "reviews", "golden_set.csv")

        df = pd.read_csv(review_data_path, encoding="iso-8859-1")

        shutil.rmtree(task_data_path)

        df["text"] = df.reviewText

        label_columns = ["classification", "class1", "class2", "class3", "class4", "class5"]

        unique_values = set()
        for column in label_columns:
            unique_values.update(df[column].unique())

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

    def get_di_sorbo_2017(self):

        task_data_path = os.path.join(self.data_path, "di_sorbo_2017")
        # from https://www.merlin.uzh.ch/contributionDocument/download/9373
        # What Would Users Change in My App? Summarizing App Reviews for Recommending Software Changes
        zip_file_path = os.path.join(task_data_path, "SURF-SURF-v.1.0.zip")
        if not os.path.exists(zip_file_path):
            r = requests.get("https://zenodo.org/record/47323/files/SURF-SURF-v.1.0.zip?download=1")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=task_data_path)

        file_dir = os.path.join(task_data_path, "panichella-SURF-29332ec", "SURF_replication_package", "Experiment I", "summaries")

        def add_data_to_df(data, df):
            soup = BeautifulSoup(data, 'html.parser')

            label_upper_element = [x for x in soup.find_all("sup")]
            text_list = [x.findNext('a').text for x in label_upper_element]
            label_list = [x.find('b').text for x in label_upper_element]

            full_review_df = pd.DataFrame({"text": text_list, "label": label_list})

            if df is None:
                df = full_review_df
            else:
                df = df.append(full_review_df)

            return df

        # Cycle through all the app files and add the review text and labels to the overall dataframe
        all_review_df = None
        for file_name in os.listdir(file_dir):
            with open(os.path.join(file_dir, file_name), "rb") as f:
                data = f.read()
                all_review_df = add_data_to_df(data, all_review_df)

        train_and_val = all_review_df.sample(n=438, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = all_review_df.drop(train_and_val.index)

        return train, val, test

    def get_guzman_2015(self):
        df = pd.read_csv("https://ase.in.tum.de/lehrstuhl_1/images/publications/Emitza_Guzman_Ortega/truthset.tsv",
                         sep="\t", names=[0, "label", 2, "app", 4, "text"])

        int_to_str_label_map = {
            5: "Praise",
            3: "Feature shortcoming",
            1: "Bug report",
            2: "Feature strength",
            7: "Usage scenario",
            4: "User request",
            6: "Complaint",
            8: "Noise"
        }
        df["label"] = df.label.apply(lambda x: int_to_str_label_map[x])

        int_to_app_name_map = {
            6: "Picsart",
            8: "Whatsapp",
            7: "Pininterest",
            1: "Angrybirds",
            3: "Evernote",
            5: "Tripadvisor",
            2: "Dropbox"
        }

        df["app"] = df.app.apply(lambda x: int_to_app_name_map[x])

        train_and_val = df.sample(frac=0.8, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)
        return train, val, test
