def get_dataset_from_df(train_df, valid_df, test_df, params):

    tokenizer = AutoTokenizer.from_pretrained(params.lm_model_name)

    def get_ids_from_text_list(df):
      return df.text.apply(partial(tokenizer.encode, max_length=params.max_length, pad_to_max_length=True))

    X_train = get_ids_from_text_list(train_df)
    X_valid = get_ids_from_text_list(valid_df)
    X_test = get_ids_from_text_list(test_df)

    def get_label_list(df):
      categorical_data = pd.Categorical(df.label)

      inv_code_map = dict(enumerate(categorical_data.categories))

      code_map = {v:k for k, v in inv_code_map.items()}

      return pd.Series(categorical_data.codes, index=df.index), code_map

    def apply_label_list(df, code_map):
      return df.label.apply(lambda x: code_map[x])

    y_train, code_map = get_label_list(train_df)
    y_valid = apply_label_list(valid_df, code_map)
    y_test = apply_label_list(test_df, code_map)

    def create_dataset(X, y, batch_size):
      X_tensor = torch.LongTensor(np.stack(X.values))
      y_tensor = torch.LongTensor(np.stack(y.values))
      return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    train_data = create_dataset(X_train, y_train, batch_size=params.batch_size_train)
    valid_data = create_dataset(X_valid, y_valid, batch_size=params.batch_size_eval)
    test_data = create_dataset(X_test, y_test, batch_size=params.batch_size_eval)

    return train_data, valid_data, test_data
