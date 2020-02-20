
def get_cross_validated_df(df, n_splits, random_state):
    kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_list = []

    for train_split, test_split in kf.split(df.index):
        fold_list.append(test_split)

    t_v_t_folds = []

    for fold_num in range(len(fold_list)):
        current_split_idx = fold_num
        next_split_idx = fold_num + 1 if fold_num + 1 < len(fold_list) else 0

        train_idx = list(range(len(fold_list)))
        train_idx.remove(current_split_idx)
        train_idx.remove(next_split_idx)

        training_index = [item for sublist in [fold_list[tr_idx] for tr_idx in train_idx] for item in sublist]
        valid_index = fold_list[current_split_idx]
        test_index = fold_list[next_split_idx]

        t_v_t_folds.append((df.loc[training_index], df.loc[valid_index], df.loc[test_index]))

    return t_v_t_folds
