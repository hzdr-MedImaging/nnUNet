from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_json
from sklearn.model_selection import KFold
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


def generate_balanced_split(preprocessed_dataset_folder: str, seed=12345, n_splits=5, group_mult=2) -> List[dict[str, List[str]]]:
    dataset = nnUNetDataset(preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=None)

    group_size = n_splits * group_mult
    id_dict = {name: name.split("_")[0] for name in dataset.keys()}
    id_inv = {}
    for k, v in id_dict.items():
        if v not in id_inv:
            id_inv[v] = []
        id_inv[v].append(k)
    files_dict = {k: [dataset[i]["data_file"] for i in v] for k, v in id_inv.items()}

    vols = {k: [np.sum(np.load(file[:-4] + ".npz", 'r')["seg"] > 0) for file in v] for k, v in files_dict.items()}
    vols_len = {k: len(v) for k, v in vols.items()}
    len_values, len_counts = np.unique(list(vols_len.values()), return_counts=True)
    splits = [{"train": [], "val": []} for i in range(n_splits)]
    reverse = False
    # first fill the folds with triples, then doubles, then singles
    for val in reversed(len_values):
        # take only triples/doubles/singles, etc. and average the volumes within each entity
        sorted_vols = {k: np.average(v) for k, v in vols.items() if len(v) == val}
        # sort by volume (descending), so we distribute the biggest volumes first
        sorted_keys, sorted_ids = zip(*sorted(zip(sorted_vols.values(), sorted_vols.keys()), reverse=True))
        # make groups of group_size with approximately equal volume
        split_ids = [sorted_ids[i:i + group_size] for i in range(0, len(sorted_ids), group_size)]
        # split groups into folds
        folds = [
            [
                np.array(split_ids[i])[test_idx] for train_idx, test_idx in
                KFold(n_splits=n_splits, shuffle=True, random_state=seed + i).split(split_ids[i])
            ] for i in range(len(split_ids))
        ]
        # glue the folds together
        folds_test = [np.concatenate([folds[j][i] for j in range(len(folds))]) for i in range(n_splits)]
        # fill the splits from the folds of
        for i in range(n_splits):
            idx = i if not reverse else n_splits - 1 - i
            splits[idx]["val"].extend(np.concatenate([id_inv[v] for v in folds_test[i]]))
        reverse = not reverse
    # now the split is filled in the way that there are singles with small volumes on top. We can attempt to reshuffle
    # the few of them to make the splits of the approximately equal size
    val_lens = [len(splits[i]["val"]) for i in range(n_splits)]
    maxdiff = max(val_lens) - min(val_lens)
    # check if it is safe to reshuffle
    if len_values[0] == 1 and len_counts[0] > maxdiff * n_splits:
        while maxdiff > 1:
            item = splits[np.argmax(val_lens)]["val"].pop()
            splits[np.argmin(val_lens)]["val"].append(item)
            val_lens = [len(splits[i]["val"]) for i in range(n_splits)]
            maxdiff = max(val_lens) - min(val_lens)

    for i in range(n_splits):
        splits[i]["train"] = list(set(id_dict.keys()) - set(splits[i]["val"]))

    # val_vol_list = [[np.sum(np.load(preprocessed_dataset_folder + "/" + file + "_seg.npy", 'r') > 0) for file in splits[i]["val"]]
    #             for i in range(n_splits)]

    return splits

if __name__ == "__main__":
    preprocessed_dataset_folder = "/pet/projekte/ai/nnUnet/preprocessed/Dataset048_Lymphoma_r13_balanced/nnUNetPlans_3d_fullres"
    splits = generate_balanced_split(preprocessed_dataset_folder, seed=12345, n_splits=5)
    # save_json(splits, "/pet/projekte/ai/nnUnet/preprocessed/Dataset048_Lymphoma_r13_balanced/splits_final.json")