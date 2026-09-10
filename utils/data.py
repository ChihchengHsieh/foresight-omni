import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from typing import List


def get_dataloader_g(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def collate_fn(batch):
    return tuple(zip(*batch))


def get_split_string(
    train_idxs: List[int], val_idxs: List[int], test_idxs: List[int], idx: int
) -> str:

    if idx in train_idxs:
        return "train"

    if idx in val_idxs:
        return "val"

    if idx in test_idxs:
        return "test"

    raise Exception(f"Index {idx} not in any split.")


def get_split_list(
    split_len: int, train_portion=0.8, val_portion=0.5, seed=0
) -> List[str]:

    split_idxs = list(range(split_len))
    print("Training val splitting")

    train_idxs, val_test_idxs = train_test_split(
        split_idxs, train_size=train_portion, random_state=seed, shuffle=True
    )

    print("Val test splitting")
    val_idxs, test_idxs = train_test_split(
        val_test_idxs, train_size=val_portion, random_state=seed, shuffle=True
    )

    print("Getting splitting string.")
    split_str_list = [""] * split_len
    for idx in train_idxs:
        split_str_list[idx] = "train"
    for idx in val_idxs:
        split_str_list[idx] = "val"
    for idx in test_idxs:
        split_str_list[idx] = "test"

    return split_str_list


def split_by_patient(df, patient_col, train_portion=0.8, val_portion=0.5, seed=0):
    groups = df[patient_col]

    # Initialize split column
    df['split'] = None

    # Step 1: Split train (80%) vs non-train (20%)
    gss = GroupShuffleSplit(n_splits=1, test_size=1-train_portion, random_state=seed)
    train_idx, non_train_idx = next(gss.split(df, groups=groups))
    df.loc[df.index[train_idx], 'split'] = 'train'

    # Step 2: Split non-train (20%) equally into val and test (50-50)
    non_train_df = df.loc[df.index[non_train_idx]]
    non_train_groups = non_train_df[patient_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=1-val_portion, random_state=seed)
    val_idx, test_idx = next(gss.split(non_train_df, groups=non_train_groups))

    df.loc[non_train_df.index[val_idx], 'split'] = 'val'
    df.loc[non_train_df.index[test_idx], 'split'] = 'test'

    return df