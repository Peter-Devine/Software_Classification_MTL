def get_NB_baseline_eval_metrics(self, train_df, test_df):

    def _get_words_from_dataset(dataset):
        """Return a set of all words in a dataset.

        :param dataset: A list of tuples of the form ``(words, label)`` where
            ``words`` is either a string of a list of tokens.
        """

        # Words may be either a string or a list of tokens. Return an iterator
        # of tokens accordingly
        def tokenize(words):
            return word_tokenize(words, include_punc=False)

        all_words = chain.from_iterable(tokenize(words.lower()) for words in dataset)
        return set(all_words)

    train_word_features = _get_words_from_dataset(train_df.text)

    def get_bow(df):

        CUSTOM_STOPWORDS = ['i', 'me','up','my', 'myself', 'we', 'our', 'ours',
                            'ourselves', 'you', 'your', 'yours','yourself', 'yourselves',
                            'he', 'him', 'his', 'himself', 'she', 'her', 'hers' ,'herself',
                            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                            'themselves' ,'am', 'is', 'are','a', 'an', 'the', 'and','in',
                            'out', 'on','up','down', 's', 't']

        tokenized_words = df.text.apply(nltk.word_tokenize)
        tokenized_words = tokenized_words.apply(lambda x: [word.lower() for word in x])
        tokenized_words = tokenized_words.apply(lambda x: [word for word in x if not word in CUSTOM_STOPWORDS])



        bow_features_lambda = lambda x: dict(((u'contains({0})'.format(word), (word in x)) for word in train_word_features))
        bow_features = tokenized_words.apply(bow_features_lambda)

        return bow_features

    baseline_X_train = get_bow(train_df)
    baseline_X_test = get_bow(test_df)

    train_baseline_data = list(zip(baseline_X_train, train_df.label))
    test_baseline_data = list(zip(baseline_X_test, test_df.label))

    NB_cls = nltk.NaiveBayesClassifier.train(train_baseline_data)

    preds = [NB_cls.classify(test_baseline_datum[0]) for test_baseline_datum in test_baseline_data]
    golds = [test_baseline_datum[1] for test_baseline_datum in test_baseline_data]

    label_classes = train_df.label.unique()
    f1_scores = sklearn.metrics.f1_score(golds, preds, average=None, labels=label_classes)
    prec_scores = sklearn.metrics.precision_score(golds, preds, average=None, labels=label_classes)
    recall_scores = sklearn.metrics.recall_score(golds, preds, average=None, labels=label_classes)
    accuracy = sklearn.metrics.accuracy_score(golds, preds)

    return {
        "label_classes": label_classes,
        "f1_scores": f1_scores,
        "prec_scores": prec_scores,
        "recall_scores": recall_scores,
        "accuracy": accuracy
    }
