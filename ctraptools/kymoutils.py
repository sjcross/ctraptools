from ctraptools import fileutils as fu
from ctraptools import imageutils as iu
from lumicks import pylake

import numpy as np
import os
import tifffile

def save_kymo(kymo, output_filename, output_range=None):
    # Getting image from kymo
    image = kymo.get_image()
            
    # Applying range normalisation
    image = iu.apply_normalisation(image, output_range)

    # Getting calibration
    time_res = 1/kymo.line_time_seconds
    spat_res = 1/kymo.pixelsize_um[1] if len(kymo.pixelsize_um) > 1 else 1/kymo.pixelsize_um[0]

    # Converting output_range to text
    or_str = iu.get_output_range_string(output_range)
    
    # Saving imag
    tifffile.imwrite(
        output_filename, 
        image.astype(np.single),
        resolution=(time_res,spat_res,"MICROMETER"),
        metadata={"OutputRange":or_str}
    )

def batch_save_kymos(input_path, output_path, output_range=None, verbose=False):
    # Creating output folder (if it doesn't already exist)
    os.makedirs(output_path) if not os.path.exists(output_path) else None      

    # Iterating over all files in the input folder
    for filename in os.listdir(input_path):

        # Processing all .h5 files (doesn't matter if they have 'Kymograph' in the name)
        if filename.endswith(".h5"):
            print("Importing "+filename)
            file = pylake.File(input_path + filename)

            # Getting kymographs and saving one-by-one
            kymos = file.kymos
            for kymo_id in kymos:
                kymo = kymos.get(kymo_id)    
                output_filename = output_path + fu.strip_ext(filename) + "_kymo" + kymo_id + ".tiff"
                save_kymo(kymo, output_filename, output_range=output_range)

def new_method():
    print('Hello world')