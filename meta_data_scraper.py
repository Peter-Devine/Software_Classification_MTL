import nltk

nltk.download("punkt")
nltk.download("stopwords")

def get_df_metadata(train_df, valid_df, test_df, PARAMS, is_multilabel):
    def get_len_of_sentence_data(df):
        num_of_words_in_each_observation = df.text.apply(lambda x: len(x.split()))
        return {
            "mean": num_of_words_in_each_observation.mean(),
            "standard_deviation": num_of_words_in_each_observation.std()
        }

    sw = nltk.corpus.stopwords.words('english')
    sw.extend([",", ".", "\"", "\'", ":", ";", "-", "/", "!", "?", "...", "(", ")"])

    def get_top_words(df):
        sentences = df.text.str.cat(sep=' ').lower()
        words = nltk.tokenize.wordpunct_tokenize(sentences)
        word_dist = nltk.FreqDist(words)

        for word in sw:
            if word in word_dist:
                word_dist.pop(word)

        return word_dist.most_common(10)

    def get_multiclass_label_count(df):
        return {
            "normalized": df.label.value_counts(normalize=True),
            "gross": df.label.value_counts(normalize=False)
        }

    def get_multilabel_label_count(df):
        dist_dict = {}
        normalized_dist_dict = {}
        size_of_dataset = df.shape[0]
        for label in [x for x in df.columns if "label_" in x]:
            gross_number = df[label].sum()
            dist_dict[label] = gross_number
            normalized_dist_dict[label] = gross_number / size_of_dataset
        return {
            "normalized": normalized_dist_dict,
            "gross": dist_dict
        }

    def get_size_data(df):
        return df.shape[0]

    def get_data_for_splits_with_fn(train, valid, test, fn):
        return {
            "train": fn(train),
            "valid": fn(valid),
            "test": fn(test)
        }

    len_data = get_data_for_splits_with_fn(train_df, valid_df, test_df, get_len_of_sentence_data)

    word_data = get_data_for_splits_with_fn(train_df, valid_df, test_df, get_top_words)

    if is_multilabel:
        label_data = get_data_for_splits_with_fn(train_df, valid_df, test_df, get_multilabel_label_count)
    else:
        label_data = get_data_for_splits_with_fn(train_df, valid_df, test_df, get_multiclass_label_count)

    size_data = get_data_for_splits_with_fn(train_df, valid_df, test_df, get_size_data)

    return {
        "top words": word_data,
        "num words in sentences": len_data,
        "label distribution": label_data,
        "data size": size_data
    }
