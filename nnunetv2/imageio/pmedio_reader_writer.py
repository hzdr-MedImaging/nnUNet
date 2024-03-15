#    Copyright 2023, Institute of Radiopharmaceutical Cancer Research, HZDR, Dresden, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import pmedio


class PmedIO(BaseReaderWriter):
    """
    Pmedio loads the images in a different order than sitk (consistent with nibabel). We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.v',
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        mheads = []
        sheads = []

        spacings_for_nnunet = []
        for f in image_fnames:
            # print(f)
            ecat_image = pmedio.read(f)
            assert ecat_image.ndim == 4, 'pmedio images supposed to be 4d'
            assert ecat_image.tdim == 1, 'only 3d images are supported by pmedio'
            mhead = ecat_image.mhead
            shead = ecat_image.shead[0]

            mheads.append(mhead)
            sheads.append(shead)

            spacings = [shead[k] for k in ("x_pixelsize", "y_pixelsize", "z_pixelsize")]

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in spacings[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(ecat_image.toarray().transpose((3, 2, 1, 0)))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same pixel spacings!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'ecat_stuff': {
                'm_header': mheads[0],
                's_header': sheads[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        images, dict = self.read_images((seg_fname, ))
        return np.round(images), dict

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0))
        seg_ecat = pmedio.MedIOImage(seg.astype(np.float32))
        # setting up proper headers
        seg_ecat.mhead = properties["ecat_stuff"]["m_header"]
        seg_ecat.shead[0] = properties["ecat_stuff"]["s_header"]
        # adding some tags for rover to identify image correctly
        seg_ecat.mhead["study_description"] = "ROI Image"
        seg_ecat.mhead["data_units"] = "NA"
        seg_ecat.shead[0]["annotation"] = "ROVER07"
        pmedio.write(seg_ecat, output_fname, overwrite=True)


# if __name__ == '__main__':
#     img_file = 'patient028_frame01_0000.nii.gz'
#     seg_file = 'patient028_frame01.nii.gz'
#
#     nibio = NibabelIO()
#     images, dct = nibio.read_images([img_file])
#     seg, dctseg = nibio.read_seg(seg_file)
#
#     nibio_r = NibabelIOWithReorient()
#     images_r, dct_r = nibio_r.read_images([img_file])
#     seg_r, dctseg_r = nibio_r.read_seg(seg_file)
#
#     nibio.write_seg(seg[0], '/home/isensee/seg_nibio.nii.gz', dctseg)
#     nibio_r.write_seg(seg_r[0], '/home/isensee/seg_nibio_r.nii.gz', dctseg_r)
#
#     s_orig = nibabel.load(seg_file).get_fdata()
#     s_nibio = nibabel.load('/home/isensee/seg_nibio.nii.gz').get_fdata()
#     s_nibio_r = nibabel.load('/home/isensee/seg_nibio_r.nii.gz').get_fdata()
