import glob
import nibabel as nib
import numpy as np

input_dir = "/pet/projekte/ai/nnUnet/raw/Dataset012_HNC/labelsTs"

files = glob.glob(input_dir + "/*.nii")
print("Running conversion of " + len(files).__str__() + " files")
for file in files:
    nifti_in = nib.load(file)
    new_data = np.copy(nifti_in.get_fdata())
    new_dtype = np.int8
    new_data = new_data.round().astype(new_dtype)
    nifti_in.set_data_dtype(new_dtype)
    nifti_out = nib.Nifti1Image(new_data, nifti_in.affine, header=nifti_in.header, dtype="int8")
    nib.save(nifti_out, file)
print("Conversion complete!")
