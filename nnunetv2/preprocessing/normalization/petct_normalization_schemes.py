import numpy as np
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization


class CTWindowNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    lower_bound = None
    upper_bound = None

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= lower_bound
        image /= upper_bound - lower_bound
        return image


class CTSoftTissueNormalization(CTWindowNormalization):
    lower_bound = -150
    upper_bound = 150


class CTLungNormalization(CTWindowNormalization):
    lower_bound = -900
    upper_bound = 150


class PETTopQNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        quantile_low, quantile_high = np.quantile(image, (0, 0.995))
        quantile_range = np.clip(quantile_high - quantile_low, a_min=1e-8, a_max=None)
        image = (image - quantile_low) / quantile_range
        image = 2 / (1 + np.exp(-image)) - 1
        image /= image.max()
        return image


class PETQNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        quantile_low, quantile_high = np.quantile(image, (0.005, 0.995))
        quantile_range = np.clip(quantile_high - quantile_low, a_min=1e-8, a_max=None)
        quantile_center = (quantile_high + quantile_low) / 2
        # map q_low and q_high to -1 and 1, respectively
        image = 2 * (image - quantile_center) / quantile_range
        # sigmoid for range compression
        image = 2 / (1 + np.exp(-image))  # range: [0, 2]
        image -= 1  # range: [-1, 1]
        return image


class PETQZNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        quantile_low, quantile_high = np.quantile(image, (0.005, 0.995))
        quantile_range = np.clip(quantile_high - quantile_low, a_min=1e-8, a_max=None)
        quantile_center = (quantile_high + quantile_low) / 2
        # map q_low and q_high to -1 and 1, respectively
        image = 2 * (image - quantile_center) / quantile_range
        # sigmoid for range compression
        image = 1 / (1 + np.exp(-image))

        # zscore normalization
        mean = image.mean()
        std = image.std()
        image -= mean
        image /= (max(std, 1e-8))
        return image

