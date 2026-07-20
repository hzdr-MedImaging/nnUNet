import math
import torch
import napari
import numpy as np
from typing import Union, List, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox

from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.pmedio_reader_writer import PmedIO


sample_images = ("0000.v",
                 "0001.v")
sample_model_dir = ""
checkpoint_name = 'checkpoint_final.pth'


def minmax_norm(data):
    assert len(data.shape) == 4, "Only 4D data are supported"
    mins = np.min(data, axis=(1, 2, 3), keepdims=True)
    maxs = np.max(data, axis=(1, 2, 3), keepdims=True)
    return (data - mins) / (maxs - mins)


def get_attention(image_files: Union[List[str], Tuple[str, ...]],
                  model_training_output_dir: str,
                  fold: Union[int, str] = 0,
                  checkpoint_name: str = 'checkpoint_final.pth'
                  ):
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir,
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
        overwrites=dict(save_attention=True)
    )

    # for name, param in predictor.network.named_parameters():
    #     if name.find("softmax_scale") > -1:
    #         print(name, ": ", param.data)

    reader = PmedIO()
    images, props = reader.read_images(image_fnames=image_files)
    # prepocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    # images, _ = prepocessor.run_case_npy(data=images, seg=None, properties=props, plans_manager=predictor.plans_manager,
    #                                   configuration_manager=predictor.configuration_manager,
    #                                   dataset_json=predictor.dataset_json)

    mins = images.min(axis=(1,2,3), keepdims=True)
    images = images - mins

    original_spacing = props['spacing']
    target_spacing = [predictor.configuration_manager.spacing[i] for i in predictor.plans_manager.transpose_backward]
    patch_size = [predictor.configuration_manager.patch_size[i] for i in predictor.plans_manager.transpose_backward]
    patch_size_resamp = compute_new_shape(patch_size, target_spacing, original_spacing)
    patch_size_resamp = tuple(patch_size_resamp)

    data = pad_nd_image(images, patch_size_resamp)

    shifts = [int((data.shape[1:][i] - patch_size_resamp[i]) / 2) for i in range(len(data.shape[1:]))]
    bbox = [[shifts[i], patch_size_resamp[i] + shifts[i]] for i in range(len(patch_size_resamp))]
    bbox = [[0, data.shape[0]]] + bbox
    data = crop_to_bbox(data, bbox)
    data = data + mins

    preds = predictor.predict_single_npy_array(data, props)
    attention_list = predictor.network.interconnect.get_all_attention_maps()

    for name, param in predictor.network.named_parameters():
        if name.find("softmax_scale") > -1:
            print(name, ": ", param.data)

    # for normalization. We do it after to not preprocess twice
    # prepocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    # data, _ = prepocessor.run_case_npy(data=data, seg=None, properties=props, plans_manager=predictor.plans_manager,
    #                                   configuration_manager=predictor.configuration_manager,
    #                                   dataset_json=predictor.dataset_json)

    return data, preds, attention_list


def get_attention_volume(rel_coords: Union[List[float], Tuple[float]],
                         attention_map: torch.Tensor,
                         target_shape: Union[List[int], Tuple[int]]):
    attention_map = attention_map.squeeze()
    init_shape = attention_map.shape
    abs_coords = [math.floor(p * s) for p, s in zip(rel_coords, init_shape)]
    vol = attention_map[None, None, abs_coords[0], abs_coords[1], abs_coords[2], :]
    return torch.nn.functional.interpolate(vol, size=target_shape, mode='nearest-exact').squeeze()


class AttentionVisualizer:
    def __init__(self,
                 data: torch.Tensor,
                 preds: torch.Tensor,
                 attention_list: torch.Tensor):
        self.data = minmax_norm(data)
        self.preds = preds
        self.attention_list = attention_list
        self.image_shape = preds.shape

        self.viewer = napari.Viewer()
        image_layer = self.viewer.add_image(self.data, contrast_limits=(0, 1))
        label_layer = self.viewer.add_labels(self.preds)

        self.head_slider = QSlider(Qt.Horizontal)
        self.head_slider.setMinimum(0)
        self.head_slider.setMaximum(len(attention_list[0]) - 1)
        self.head_slider.setSingleStep(1)

        self.head_slider.valueChanged[int].connect(
            lambda value=self.head_slider: self.display_attention(update=True)
        )

        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setMinimum(0)
        self.level_slider.setMaximum(len(attention_list) - 1)
        self.level_slider.setSingleStep(1)

        self.level_slider.valueChanged[int].connect(
            lambda value=self.level_slider: self.display_attention(update=True)
        )

        self.viewer.window.add_dock_widget(self.head_slider, name='Attention head selection', area='left')
        self.viewer.window.add_dock_widget(self.level_slider, name='Attention level selection', area='left')

        self.pos = None

        @self.viewer.mouse_drag_callbacks.append
        def get_event(viewer, event):
            pos = [round(p) for p in event.position[slice(1, 4)]]
            checks = [(p >= 0) & (p < s) for p, s in zip(pos, self.image_shape)]
            if all(checks):
                if 'Control' in event.modifiers:
                    self.pos = pos
                    self.display_attention()

    def display_attention(self, update=False):
        if self.pos is None:
            return

        if 'attention' in self.viewer.layers:
            self.viewer.layers.remove('attention')
        elif update:
            return

        self.viewer.status = "Coord: " + str(self.pos) + "; Head " + str(self.head_slider.value()) + "; Level " + str(self.level_slider.value())
        relative_coord = [p / s for p, s in zip(self.pos, self.image_shape)]
        attention_vol = get_attention_volume(relative_coord,
                                             self.attention_list[self.level_slider.value()][self.head_slider.value()],
                                             self.image_shape)
        attention_layer = self.viewer.add_image(attention_vol, opacity=0.5, blending='translucent',
                                           colormap='red', name='attention', contrast_limits=(0, attention_vol.max()))

    def run(self):
        napari.run()


if __name__ == '__main__':
    data, preds, attention_list = get_attention(sample_images, sample_model_dir, checkpoint_name=checkpoint_name)
    vis = AttentionVisualizer(data, preds, attention_list)
    vis.run()

