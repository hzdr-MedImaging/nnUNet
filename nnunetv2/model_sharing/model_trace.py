import torch

from nnunetv2.utilities.file_path_utilities import *
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

def trace_model(dataset_name_or_id: Union[int, str],
                            configurations: Tuple[str] = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                            trainer: str = 'nnUNetTrainer',
                            plans_identifier: str = 'nnUNetPlans',
                            folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
                            save_checkpoints: Tuple[str, ...] = ('checkpoint_final.pth',)) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    for c in configurations:
        print(f"Configuration {c}")
        trainer_output_dir = get_output_folder(dataset_name, trainer, plans_identifier, c)

        if not isdir(trainer_output_dir):
            raise RuntimeError(f"{dataset_name} is missing the trained model of configuration {c}")

        expected_fold_folder = [f"fold_{i}" if i != 'all' else 'fold_all' for i in folds]
        assert all([isdir(join(trainer_output_dir, i)) for i in expected_fold_folder]), \
            f"not all requested folds are present; {dataset_name} {c}; requested folds: {folds}"

        assert isfile(join(trainer_output_dir, "plans.json")), f"plans.json missing, {dataset_name} {c}"

        # plans
        plans_file = join(trainer_output_dir, "plans.json")
        # fingerprint
        fingerprint_file = join(trainer_output_dir, "dataset_fingerprint.json")
        # dataset
        dataset_file = join(trainer_output_dir, "dataset.json")

        plans = load_json(plans_file)
        dataset = load_json(dataset_file)

        plans_manager = PlansManager(plans_file)
        configuration_manager = plans_manager.get_configuration(c)

        in_channels = len(dataset["channel_names"])
        out_channels = len(dataset["labels"])
        model = get_network_from_plans(configuration_manager.network_arch_class_name,
                                         configuration_manager.network_arch_init_kwargs,
                                         configuration_manager.network_arch_init_kwargs_req_import,
                                         in_channels,
                                         out_channels,
                                         True)

        for fold_folder in expected_fold_folder:
            print(f"Exporting {fold_folder}")
            # debug.json, does not exist yet
            # source_file = join(trainer_output_dir, fold_folder, "debug.json")
            # if isfile(source_file):
            #     zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

            # all requested checkpoints
            for chk in save_checkpoints:
                chk_file = join(trainer_output_dir, fold_folder, chk)
                checkpoint = torch.load(chk_file, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint["network_weights"])
                model.eval()

                example_shape = [configuration_manager.batch_size, in_channels, ] + configuration_manager.patch_size
                example = torch.rand(example_shape, device=torch.device('cpu'))
                # script = torch.jit.script(model)
                script = torch.jit.trace(model, example)
                out_filename = chk.replace(".pth", "_traced.pt")
                script_file = join(trainer_output_dir, fold_folder, out_filename)
                script.save(script_file)

    print('Done')


if __name__ == '__main__':
    trace_model(31,
                trainer="nnUNetTrainer_noSmooth",
                plans_identifier="nnUNetResEncUNetLPlans",
                configurations=("3d_fullres",)
                # folds=("all",)
                )
