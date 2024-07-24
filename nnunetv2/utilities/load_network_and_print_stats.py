import json

import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.paths import nnUNet_results

from batchgenerators.utilities.file_and_folder_operations import load_json, join

from torchinfo import summary


def load_model(dataset, train_id, configuration=None):
    nnUnetDir = nnUNet_results
    if nnUnetDir is None:
        nnUnetDir = "/pet/projekte/ai/nnUnet/results"
    config_path = join(nnUnetDir, dataset, train_id)
    plans_file = join(config_path, "plans.json")
    dataset_file = join(config_path, "dataset.json")

    plans = load_json(plans_file)
    dataset = load_json(dataset_file)

    if configuration is None:
        # attempt to recover configuration name from train_id
        configuration = train_id.split("__")[-1]

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    in_channels = len(dataset["channel_names"])
    out_channels = len(dataset["labels"])

    network = get_network_from_plans(configuration_manager.network_arch_class_name,
                                     configuration_manager.network_arch_init_kwargs,
                                     configuration_manager.network_arch_init_kwargs_req_import,
                                     in_channels,
                                     out_channels,
                                     True)

    return network


def print_summary(dataset, train_id, configuration=None):
    nnUnetDir = nnUNet_results
    if nnUnetDir is None:
        nnUnetDir = "/pet/projekte/ai/nnUnet/results"
    config_path = join(nnUnetDir, dataset, train_id)
    plans_file = join(config_path, "plans.json")
    dataset_file = join(config_path, "dataset.json")

    plans = load_json(plans_file)
    dataset = load_json(dataset_file)

    if configuration is None:
        # attempt to recover configuration name from train_id
        configuration = train_id.split("__")[-1]

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    in_channels = len(dataset["channel_names"])
    out_channels = len(dataset["labels"])
    network = get_network_from_plans(configuration_manager.network_arch_class_name,
                    configuration_manager.network_arch_init_kwargs,
                    configuration_manager.network_arch_init_kwargs_req_import,
                    in_channels,
                    out_channels,
                    True)

    example_shape = [configuration_manager.batch_size, in_channels,] + configuration_manager.patch_size
    print("Decoder:")
    summary(network, input_size=example_shape, col_names=("input_size", "output_size", "num_params"))
    print()
    print("Encoder:")
    summary(network.encoder, input_size=example_shape,  col_names=("input_size", "output_size", "num_params"))
    print()
    print("First two encoder blocks:")
    s = summary(network.encoder.stages[0], input_size=example_shape,  col_names=("input_size", "output_size", "num_params"))
    print(s)
    summary(network.encoder.stages[1], input_size=s.summary_list[-1].output_size,  col_names=("input_size", "output_size", "num_params"))

    print()
    print("Skip connections:")
    skips = network.encoder.forward(torch.rand(example_shape))
    for skip in skips:
        print(list(skip.shape))


if __name__ == '__main__':
    print_summary("Dataset021_spheroids", "nnUNetTrainer__nnUNetPlans__2d")