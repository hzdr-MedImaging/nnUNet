from pathlib import Path
import json
import os


def split_fold_files(
    input_dir,
    output_dir,
    splits_file,
    file_suffixes=None,
    use_symlink=True,
):
    """
    Link files from input_dir into fold subdirectories according to
    nnUNet splits_final.json test lists.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing source files.

    output_dir : str or Path
        Output directory. Creates:
            output_dir/fold_0/
            output_dir/fold_1/
            ...

    splits_file : str or Path
        Path to nnUNet splits_final.json.

    file_suffixes : str, optional
        Optional suffixes to append when searching files.
        Example:
            ["_0000.nii.gz", "_0001.nii.gz"]

        If None, matches any file beginning with case id.

    use_symlink : bool
        If True, create symbolic links.
        If False, create hard links.

    Notes
    -----
    Expects nnUNet split entries like:

    [
        {
            "train": [...],
            "val": [...]
        },
        ...
    ]

    Validation cases are treated as test cases for each fold.
    """

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    splits_file = Path(splits_file).resolve()

    with open(splits_file, "r") as f:
        splits = json.load(f)

    for fold_idx, split in enumerate(splits):

        # nnUNet uses "val" as fold test set
        test_cases = split["val"]

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for case_id in test_cases:

            if file_suffixes is not None:
                matches = [list(input_dir.glob(f"{case_id}{file_suffix}")) for file_suffix in file_suffixes]
                matches = [x for submatch in matches for x in submatch]
            else:
                matches = list(input_dir.glob(f"{case_id}*"))

            if len(matches) == 0:
                print(f"[WARNING] No files found for: {case_id}")
                continue

            for src in matches:

                dst = fold_dir / src.name

                if dst.exists():
                    continue

                if use_symlink:
                    os.symlink(src, dst)
                else:
                    os.link(src, dst)

        print(f"Fold {fold_idx}: linked {len(test_cases)} cases")


# Example usage
if __name__ == "__main__":

    split_fold_files(
        input_dir="/pet/projekte/ai/delineation/lymphoma/data/r13rdr/",
        output_dir="/pet/projekte/ai/delineation/lymphoma/data/r13rdr/folds/",
        splits_file="/pet/projekte/ai/delineation/lymphoma/data/r13rdr/folds/splits_final.json",
        file_suffixes=["_0000.v", "_0001.v"],
        use_symlink=True,
    )