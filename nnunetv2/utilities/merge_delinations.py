import pathlib
import sys
import getopt
import nibabel as nib
import numpy as np


def print_help():
    def print_option(opt, descr):
        print(f"    {opt:16s}{descr}")

    print("merge_delineations - merges multiple binary delineations into one using a specified strategy")
    print("Copyright (c) 2024-2025 Pavel Nikulin, Jens Maus, www.hzdr.de")
    print()
    print('Usage: python merge_delineations [-s strategy] [-h] output_dir input_dirs')
    print()
    print('Positional arguments:')
    print_option("output_dir", "Output directory name")
    print_option("input_dirs", "List of input directory names. Should contain NIfTis as .nii.gz files. All filenames in the input directories have to match")
    print()
    print('Optional arguments:')
    print_option("-s", "Merge strategy: u - union (default), i - intersection, m - majority voting (strict)")
    print_option("-h", "Displays this help")

    sys.exit()

def main():
    strategy = ''

    # dealing with command line and checking inputs
    if len(sys.argv) < 2:
        print_help()

    # parse command line
    argsv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argsv, "hs:")
    except getopt.GetoptError as e:
        print("Incorrect input configuration:", e.msg)
        sys.exit()
    if len(args) == 0:
        sys.exit("No input/output directories specified")
    if len(args) == 1:
        sys.exit("No input directories are specified")
    if len(args) == 2:
        sys.exit("Only one input directory is specified")
    output_dir = args[0]
    input_dirs = args[1:]

    # set options
    for opt, arg in opts:
        if opt == '-h':
            print_help()
        elif opt == "-s":
            strategy = arg

    # check options
    if len(strategy) == 0:
        strategy = "u"
    if strategy not in ("u", "i", "m"):
        sys.exit("Unrecognized merging strategy")

    if not all([pathlib.Path(input_dir).is_dir() for input_dir in input_dirs]):
        sys.exit("All input targets have to be directories")
    if pathlib.Path(output_dir).exists() and not pathlib.Path(output_dir).is_dir():
        sys.exit("Output target has to be a directory")
    # create output dir
    if not pathlib.Path(output_dir).exists():
        if pathlib.Path(output_dir).parent.exists():
            pathlib.Path(output_dir).mkdir()
        else:
            sys.exit("Output directory cannot be created: parent directory does not exist")

    print("Merging delineations")
    # show options
    print('Input directories: ', input_dirs)
    print('Output directory:',   output_dir)
    #print('Merging strategy:',   strategy)

    input_files = []
    input_files = [list(pathlib.Path(input_dir).glob("*.nii.gz")) for input_dir in input_dirs]
    if len(set([len(l) for l in input_files])) > 1:
        sys.exit("Number of nifties in input folders do not match")

    input_basenames = [[input_file.name for input_file in input_folder] for input_folder in input_files]
    __ = [input_dir.sort() for input_dir in input_basenames] #sort lists
    if input_basenames.count(input_basenames[0]) != len(input_basenames):
        sys.exit("Files in input directories do not match")

    for file_name in input_basenames[0]:
        files_in = [pathlib.Path(input_dir).joinpath(file_name) for input_dir in input_dirs]
        file_out = pathlib.Path(output_dir).joinpath(file_name)

        nifties_in = [nib.load(file) for file in files_in]
        vols_in = [nifti.get_fdata() for nifti in nifties_in]

        if strategy == "u":
            result = np.logical_or.reduce(vols_in)
        if strategy == "i":
            result = np.logical_and.reduce(vols_in)
        if strategy == "m":
            result = np.average(vols_in, axis=0) > 0.5
        nifti_out = nib.Nifti1Image(result.astype(np.uint8), affine=nifties_in[0].affine, header=nifties_in[0].header)
        nib.save(nifti_out, file_out)

if __name__ == "__main__":
    main()