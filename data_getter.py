DATASET_TO_DOWNLOADER = {
    "maalej_2015": get_maalej_2015
}

def get_selected_datasets(names_of_datasets):
    retreived_datasets = {}
    for dataset_name in names_of_datasets:
        retreived_dataset = DATASET_TO_DOWNLOADER[dataset_name]()
        retreived_datasets[dataset_name] = retreived_dataset

    return retreived_datasets

def get_maalej_2015():

    import requests, zipfile, io

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

    return df
