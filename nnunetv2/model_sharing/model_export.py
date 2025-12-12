import tempfile
import zipfile
import torch

from nnunetv2.utilities.file_path_utilities import *
from batchgenerators.utilities.file_and_folder_operations import load_json

def get_reader_and_ext(file_format: str) -> Tuple[str, str]:
    if file_format == "nifti":
        reader = "NibabelIOWithReorient"
        extension = ".nii.gz"
    else:
        raise ValueError("Unsupported new file format")
    return reader, extension

def convert_vsize(vsize: List[float],
                  old_reader: str,
                  new_reader: str) -> List[float]:
    if old_reader == "PmedIO" and new_reader == "NibabelIOWithReorient":
        scale = 10
    elif old_reader == "NibabelIOWithReorient" and new_reader == "PmedIO":
        scale = 0.1
    else:
        raise ValueError("Unsupported pair of readers for conversion")
    vsize = [v * scale for v in vsize]
    return vsize


def export_pretrained_model(dataset_name_or_id: Union[int, str], output_file: str,
                            configurations: Tuple[str] = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                            trainer: str = 'nnUNetTrainer',
                            plans_identifier: str = 'nnUNetPlans',
                            folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
                            strict: bool = True,
                            stripped: bool = False,
                            save_checkpoints: Tuple[str, ...] = ('checkpoint_final.pth',),
                            export_crossval_predictions: bool = False,
                            new_dataset_name: str = None,
                            new_trainer: str = None,
                            new_file_format: str = None) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    if new_dataset_name is None:
        new_dataset_name = dataset_name
    if new_trainer is None:
        new_trainer = trainer

    if new_file_format is not None:
        new_reader, new_extension = get_reader_and_ext(new_file_format)

    def arcpath(source_file: str):
        rp = os.path.relpath(source_file, nnUNet_results)
        rp = rp.replace(dataset_name + '/', new_dataset_name + '/')
        rp = rp.replace("/" + trainer + '__', "/" + new_trainer + '__')
        return rp

    with(zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)) as zipf:
        for c in configurations:
            print(f"Configuration {c}")
            trainer_output_dir = get_output_folder(dataset_name, trainer, plans_identifier, c)

            if not isdir(trainer_output_dir):
                if strict:
                    raise RuntimeError(f"{dataset_name} is missing the trained model of configuration {c}")
                else:
                    continue

            expected_fold_folder = [f"fold_{i}" if i != 'all' else 'fold_all' for i in folds]
            assert all([isdir(join(trainer_output_dir, i)) for i in expected_fold_folder]), \
                f"not all requested folds are present; {dataset_name} {c}; requested folds: {folds}"

            assert isfile(join(trainer_output_dir, "plans.json")), f"plans.json missing, {dataset_name} {c}"

            for fold_folder in expected_fold_folder:
                print(f"Exporting {fold_folder}")
                # debug.json, does not exist yet
                if not stripped:
                    source_file = join(trainer_output_dir, fold_folder, "debug.json")
                    if isfile(source_file):
                        zipf.write(source_file, arcpath(source_file))

                # all requested checkpoints
                for chk in save_checkpoints:
                    source_file = join(trainer_output_dir, fold_folder, chk)
                    if new_trainer is None:
                        zipf.write(source_file, arcpath(source_file))
                    else:
                        checkpoint = torch.load(source_file, map_location=torch.device('cpu'))
                        checkpoint['trainer_name'] = new_trainer
                        with tempfile.NamedTemporaryFile('wb', suffix='.pth') as chk_file:
                            torch.save(checkpoint, chk_file.name)
                            zipf.write(chk_file.name, arcpath(source_file))

                # progress.png
                source_file = join(trainer_output_dir, fold_folder, "progress.png")
                zipf.write(source_file, arcpath(source_file))

                # if it exists, network architecture.png
                source_file = join(trainer_output_dir, fold_folder, "network_architecture.pdf")
                if isfile(source_file):
                    zipf.write(source_file, arcpath(source_file))

                # validation folder with all predicted segmentations etc
                if not stripped:
                    if export_crossval_predictions:
                        source_folder = join(trainer_output_dir, fold_folder, "validation")
                        files = [i for i in subfiles(source_folder, join=False) if not i.endswith('.npz') and not i.endswith('.pkl')]
                        for f in files:
                            zipf.write(join(source_folder, f), arcpath(join(source_folder, f)))
                    # just the summary.json file from the validation
                    else:
                        source_file = join(trainer_output_dir, fold_folder, "validation", "summary.json")
                        zipf.write(source_file, arcpath(source_file))

            source_folder = join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
            if isdir(source_folder) and not stripped:
                if export_crossval_predictions:
                    source_files = subfiles(source_folder, join=True)
                else:
                    source_files = [
                        join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}', i) for i in
                        ['summary.json', 'postprocessing.pkl', 'postprocessing.json']
                    ]
                for s in source_files:
                    if isfile(s):
                        zipf.write(s, arcpath(s))

            # plans
            source_file = join(trainer_output_dir, "plans.json")
            if new_file_format is None and dataset_name == new_dataset_name and not stripped:
                zipf.write(source_file, arcpath(source_file))
            else:
                plans = load_json(source_file)
                plans['dataset_name'] = new_dataset_name
                if new_file_format is not None:
                    old_reader = plans['image_reader_writer']
                    plans['image_reader_writer'] = new_reader
                    plans['original_median_spacing_after_transp'] = convert_vsize(
                        plans['original_median_spacing_after_transp'], old_reader = old_reader, new_reader = new_reader
                    )
                    for _, config in plans['configurations'].items():
                        if 'spacing' in config:
                            config['spacing'] = convert_vsize(
                                config['spacing'], old_reader = old_reader, new_reader = new_reader
                            )
                if stripped:
                    for config_name in list(plans['configurations'].keys()):
                        if config_name not in (c,) + ('3d_lowres', '3d_fullres', '2d', '3d_cascade_fullres'):
                            del plans['configurations'][config_name]
                        elif plans['configurations'][config_name].get('inherits_from') is not None and plans['configurations'][config_name].get('inherits_from') not in ('3d_lowres', '3d_fullres', '2d', '3d_cascade_fullres'):
                            raise RuntimeError("You configuration inherits from another non-standard configuration. It will not work together with the --strip option")
                with tempfile.NamedTemporaryFile('wb', suffix='.json') as plans_file:
                    save_json(plans, plans_file.name, sort_keys=False)
                    zipf.write(plans_file.name, arcpath(source_file))

            # fingerprint
            if not stripped:
                source_file = join(trainer_output_dir, "dataset_fingerprint.json")
                zipf.write(source_file, arcpath(source_file))

            # dataset
            source_file = join(trainer_output_dir, "dataset.json")
            if new_file_format is None:
                zipf.write(source_file, arcpath(source_file))
            else:
                dataset = load_json(source_file)
                dataset['file_ending'] = new_extension
                dataset['overwrite_image_reader_writer'] = new_reader
                with tempfile.NamedTemporaryFile('wb', suffix='.json') as dataset_file:
                    save_json(dataset, dataset_file.name, sort_keys=False)
                    zipf.write(dataset_file.name, arcpath(source_file))


        ensemble_dir = join(nnUNet_results, dataset_name, 'ensembles')

        if not isdir(ensemble_dir):
            print("No ensemble directory found for task", dataset_name_or_id)
            return
        if stripped:
            return
        subd = subdirs(ensemble_dir, join=False)
                # figure out whether the models in the ensemble are all within the exported models here
        for ens in subd:
            identifiers, folds = convert_ensemble_folder_to_model_identifiers_and_folds(ens)
            ok = True
            for i in identifiers:
                tr, pl, c = convert_identifier_to_trainer_plans_config(i)
                if tr == trainer and pl == plans_identifier and c in configurations:
                    pass
                else:
                    ok = False
            if ok:
                print(f'found matching ensemble: {ens}')
                source_folder = join(ensemble_dir, ens)
                if export_crossval_predictions:
                    source_files = subfiles(source_folder, join=True)
                else:
                    source_files = [
                        join(source_folder, i) for i in
                        ['summary.json', 'postprocessing.pkl', 'postprocessing.json'] if isfile(join(source_folder, i))
                    ]
                for s in source_files:
                    zipf.write(s, arcpath(s))
        inference_information_file = join(nnUNet_results, dataset_name, 'inference_information.json')
        if isfile(inference_information_file):
            zipf.write(inference_information_file, arcpath(inference_information_file))
        inference_information_txt_file = join(nnUNet_results, dataset_name, 'inference_information.txt')
        if isfile(inference_information_txt_file):
            zipf.write(inference_information_txt_file, arcpath(inference_information_txt_file))
    print('Done')


if __name__ == '__main__':
    export_pretrained_model(2, '/home/fabian/temp/dataset2.zip', strict=False, export_crossval_predictions=True, folds=(0, ))
