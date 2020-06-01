import requests, zipfile, io, os, json, shutil, time
from io import StringIO
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup
from task import Task
from models import get_language_model

class TaskBuilder:
    def __init__(self, random_state):
        self.random_state = random_state
        self.task_dict = {
            "maalej_2016": Task(data_getter_fn=self.get_maalej_2016, is_multilabel=False),
            "maalej_bug_bin_2016": Task(data_getter_fn=self.get_bin_df_function(self.get_maalej_2016, "bug"), is_multilabel=False),
            "maalej_small_2016": Task(data_getter_fn=self.get_small_df_function(self.get_maalej_2016), is_multilabel=False),

            "williams_2017": Task(data_getter_fn=self.get_williams_2017, is_multilabel=False),
            "williams_bug_bin_2017": Task(data_getter_fn=self.get_bin_df_function(self.get_williams_2017, "bug"), is_multilabel=False),
            "williams_small_2017": Task(data_getter_fn=self.get_small_df_function(self.get_williams_2017), is_multilabel=False),

            "chen_2014_swiftkey": Task(data_getter_fn=self.get_chen_2014_swiftkey, is_multilabel=False),

            "ciurumelea_2017_fine": Task(data_getter_fn=self.get_ciurumelea_2017_fine, is_multilabel=True),
            "ciurumelea_2017_coarse": Task(data_getter_fn=self.get_ciurumelea_2017_coarse, is_multilabel=True),

            "di_sorbo_2017": Task(data_getter_fn=self.get_di_sorbo_2017, is_multilabel=False),
            "di_sorbo_bug_bin_2017": Task(data_getter_fn=self.get_bin_df_function(self.get_di_sorbo_2017, "bug"), is_multilabel=False),
            "di_sorbo_small_2017": Task(data_getter_fn=self.get_small_df_function(self.get_di_sorbo_2017), is_multilabel=False),

            "guzman_2015": Task(data_getter_fn=self.get_guzman_2015, is_multilabel=False),
            "guzman_bug_bin_2015": Task(data_getter_fn=self.get_bin_df_function(self.get_guzman_2015, "bug"), is_multilabel=False),
            "guzman_small_2015": Task(data_getter_fn=self.get_small_df_function(self.get_guzman_2015), is_multilabel=False),

            "scalabrino_2017": Task(data_getter_fn=self.get_scalabrino_2017, is_multilabel=False),
            "scalabrino_bug_bin_2017": Task(data_getter_fn=self.get_bin_df_function(self.get_scalabrino_2017, "bug"), is_multilabel=False),
            "scalabrino_small_2017": Task(data_getter_fn=self.get_small_df_function(self.get_scalabrino_2017), is_multilabel=False),

            "jha_2017": Task(data_getter_fn=self.get_jha_2017, is_multilabel=False),
            "jha_bug_bin_2017": Task(data_getter_fn=self.get_bin_df_function(self.get_jha_2017, "bug"), is_multilabel=False),
            "jha_small_2017": Task(data_getter_fn=self.get_small_df_function(self.get_jha_2017), is_multilabel=False),

            "morales_ramirez_2019": Task(data_getter_fn=self.get_morales_ramirez_2019, is_multilabel=False),

            "tizard_2019": Task(data_getter_fn=self.get_tizard_2019, is_multilabel=False),
            "tizard_bug_bin_2019": Task(data_getter_fn=self.get_bin_df_function(self.get_tizard_2019, "bug"), is_multilabel=False),
            "tizard_small_2019": Task(data_getter_fn=self.get_small_df_function(self.get_tizard_2019), is_multilabel=False),
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

    def get_bin_df_function(self, df_fn, bin_label):
        def bin_df():
            train, valid, test = df_fn()

            label_binarizer = lambda x: bin_label if bin_label.lower() in x.lower() else f"other"
            train["label"] = train["label"].apply(label_binarizer)
            valid["label"] = valid["label"].apply(label_binarizer)
            test["label"] = test["label"].apply(label_binarizer)

            return train, valid, test
        return bin_df

    def get_small_df_function(self, df_fn):
        def bin_df():
            train, valid, test = df_fn()

            train = train.sample(350, random_state=self.random_state)
            valid = valid.sample(150, random_state=self.random_state)
            test = test.sample(200, random_state=self.random_state)

            return train, valid, test
        return bin_df


    ######### INDIVIDUAL DATA GETTERS ############

    def get_maalej_2016(self):
        task_data_path = os.path.join(self.data_path, "maalej_2016")

        # Sometimes, retrieving the Maalej dataset results in not getting the zip file, which will be a rate limiting atrifact. So we simply catch any errors and sleep for 10 mins before retrying.
        try:
            # from https://mast.informatik.uni-hamburg.de/wp-content/uploads/2015/06/review_classification_preprint.pdf
            # Bug Report, Feature Request, or Simply Praise? On Automatically Classifying App Reviews
            r = requests.get("https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path = task_data_path)
        except Exception as err:
            print(f"The following error has been thrown when trying to retreive the Maalej 2016 dataset from mast.informatik.uni-hamburg.de:\n\n{err}\n\nResponse from trying to reach website was:\n{r.content}")
            print(f"Now sleeping for 10 mins and then retrying (N.B. unlimited retries)")
            time.sleep(600)
            return self.get_maalej_2016()


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
        task_data_path = os.path.join(self.data_path, "williams_2017")
        # from
        # Mining Twitter feeds for software user requirements.
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

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test

    def get_chen_2014_swiftkey(self):
        task_data_path = os.path.join(self.data_path, "chen_2014")
        # from https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3323&context=sis_research
        # AR-Miner: Mining Informative Reviews for Developers from Mobile App Marketplace
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

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test

    def get_di_sorbo_2017(self):

        task_data_path = os.path.join(self.data_path, "di_sorbo_2017")
        # from https://www.merlin.uzh.ch/contributionDocument/download/9373
        # What Would Users Change in My App? Summarizing App Reviews for Recommending Software Changes
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

        train_and_val = all_review_df.sample(frac=0.7, random_state=self.random_state)
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

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)
        return train, val, test

    def get_jha_2017(self):

        task_data_path = os.path.join(self.data_path, "jha_2017")
        # from https://www.springer.com/content/pdf/10.1007%2F978-3-319-54045-0.pdf
        # Mining User Requirements from Application Store Reviews Using Frame Semantics
        r = requests.get("http://seel.cse.lsu.edu/data/refsq17.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        def get_jha_df(filename):
            review_data_path = os.path.join(task_data_path, "refsq17", "refsq17", "BOW", filename)

            with open(review_data_path, "r") as f:
                text_data = f.read()

            df = pd.read_csv(StringIO(text_data), names=["text", "label"])

            # Strip out unnecessary ' that bookends every review
            df.text = df.text.apply(lambda x: x.strip("'"))
            # Strip the unnecessary whitespace that prepends every label
            df.label = df.label.apply(lambda x: x.strip())

            return df

        train_and_val = get_jha_df("BOW_training.txt")
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = get_jha_df("BOW_testing.txt")

        return train, val, test

    def get_scalabrino_2017(self):
        df = pd.read_csv("https://dibt.unimol.it/reports/clap/downloads/rq3-manually-classified-implemented-reviews.csv")

        df = df.rename(columns = {"body": "text", "category": "label"})

        # We take out a randomly sampled one of every label to make sure that the training dataset has one label for each class
        unique_df = df.groupby('label',as_index = False,group_keys=False).apply(lambda s: s.sample(1, random_state=self.random_state))
        df = df.drop(unique_df.index)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        train = train.append(unique_df)
        test = df.drop(train_and_val.index)

        return train, val, test

    def get_morales_ramirez_2019(self):
        task_data_path = os.path.join(self.data_path, "morales_ramirez_2019")

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        URL = "https://docs.google.com/uc?export=download"

        id_ = "1PIEAY3o1RGNiIVeASYSoKtTqRXPJxsQ6"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id_ }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id_, 'confirm' : token }
            r = session.get(URL, params = params, stream = True)

        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        csv_file_location = os.path.join(task_data_path, "FilteredComments3Classes01Nov2017.csv")

        df = pd.read_csv(csv_file_location, names=["id", "unknown", "text", "feedback_type", "label"], encoding='ISO-8859-1')

        # We append the string "_(BUG)" to all "DEFECT" labels so that we can search for the word "bug" in all labels when doing zero-shot evaluation
        df.label = df.label.apply(lambda x: f"{x}_(BUG)" if "defect" in x.lower() else x)

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test

    def get_tizard_2019(self):
        task_data_path = os.path.join(self.data_path, "tizard_2019")

        r = requests.get("https://zenodo.org/record/3340156/files/RE_Submission_17-master.zip?download=1")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=task_data_path)

        db = sqlite3.connect(os.path.join(task_data_path, "RE_Submission_17-master", "VLC_labelled_sentences_4RE.sqlite"))

        column_names = [x[1] for x in db.execute("""PRAGMA table_info('labelled_sentences');""")]

        row_data = [row for row in db.execute("SELECT * from labelled_sentences")]

        df = pd.DataFrame(row_data, columns=column_names)

        bad_label_list = [
            "and Remove Cookies Warning\xa0! ",
            "and Remove Cookies  ",
            "remove",
            "continued from previous"
        ]

        # Remove the four rows with mistake labels
        df = df.loc[~df.label.apply(lambda x: x in bad_label_list)]

        def forum_label_transformer(raw_label):
            lower_label = raw_label.lower()

            if "application guidance" in lower_label:
                return "application guidance"
            if "non-informative" in lower_label:
                return "non-informative"
            if "apparent bug" in lower_label:
                return "apparent bug"
            if "question on application" in lower_label:
                return "question on application"
            if "feature request" in lower_label:
                return "feature request"
            if "help seek" in lower_label:
                return "help seeking"
            if "user setup" in lower_label:
                return "user setup"
            if "usage" in lower_label:
                return "application usage"
            if "is-background" in lower_label:
                return "question on background"
            if "attempted solution" in lower_label:
                return "attempted solution"
            if "requesting" in lower_label:
                return "requesting more information"
            if "dispraise" in lower_label:
                return "dispraise for application"
            if "praise application" in lower_label:
                return "praise for application"
            if "acknowledgement" in lower_label:
                return "acknowledgement of problem resolution"
            if "agreeing with the problem" in lower_label:
                return "agreeing with the problem"
            if "limitation confirmation" in lower_label:
                return "limitation confirmation"
            if "bug confirmation" in lower_label:
                return "malfunction confirmation"
            if "agreeing with the request" in lower_label:
                return "agreeing with the feature request"
            return "other"

        # Map all labels to their lower-case proper label. Mark all labels outside of the literature label set as "other".
        df.label = df.label.apply(forum_label_transformer)

        df = df.rename(columns={'sentence': 'text'})

        train_and_val = df.sample(frac=0.7, random_state=self.random_state)
        train = train_and_val.sample(frac=0.7, random_state=self.random_state)
        val = train_and_val.drop(train.index)
        test = df.drop(train_and_val.index)

        return train, val, test
