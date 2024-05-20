from ctraptools.kymos.detect import gauss_1D, get_raw_profile
from ctraptools.kymos.kymo import TrackMeasures
from lumicks import pylake
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import csv
import ctraptools.utils.fileutils as fu
import ctraptools.utils.imageutils as iu
import imageio.v3 as io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import random
import re
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

def read_image(path,channel,x_range=None):
    image = io.imread(path)
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = image[:,:,channel]
        elif image.shape[1] == 3:
            image = image[:,channel,:]
        else:
            image = image[channel,:,:]

    if x_range is not None:
        image = image[:,x_range[0]:x_range[1]]

    return image

def read_image_calibration(path):
    spatial_scale = None
    spatial_units = None
    time_scale = None
    time_units = None
    
    # Checking for Lumicks metadata format
    info = io.immeta(path).get('Info')
    if 'Comment = {' in info:
        spatial_pattern = re.compile(r'Pixel size \((.+)\)\":.+([0-9]+\.[0-9]+)[^\r]')
        spatial_match = spatial_pattern.search(info)
        if spatial_match is not None:
            spatial_units = spatial_match[1]
            spatial_scale = float(spatial_match[2])
            
        time_pattern = re.compile(r'Line time \((.+)\)\":.+([0-9]+\.[0-9]+)[^\r]')
        time_match = time_pattern.search(info)
        time_scale = None
        if time_match is not None:
            time_units = time_match[1]
            time_scale = float(time_match[2])

    else:
        # Using plain tif calibration
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            
            (time_cal,time_range) = tags.get('XResolution').value
            time_scale = time_range/time_cal
            
            (spatial_cal,spatial_range) = tags.get('YResolution').value
            spatial_scale = spatial_range/spatial_cal

            ij_meta = tif.imagej_metadata
            if ij_meta is not None:
                time_units = ij_meta.get('unit')
                spatial_units = ij_meta.get('yunit')
                if spatial_units == '\\u00B5m':
                    spatial_units = "μm"
        
    return(spatial_scale,spatial_units,time_scale,time_units)

def write_change_points(tracks, filepath):
    with open(filepath+".csv", 'w', newline='') as file:
        writer = csv.writer(file)

        # Adding header row
        writer.writerow(['TrackID','Number of steps','Step positions (timpoint) ->'])

        for track in tracks.values():
            row = []
            row.append(str(track.ID))
            row.append(str(len(track.steps)))
            for step in track.steps:
                row.append(step)
            writer.writerow(row)

def write_peak_traces(tracks, filepath, extra_columns=None):
    for track in tracks.values():
        with open(filepath+"_ID"+str(track.ID)+".csv", 'w', newline='') as file:
            writer = csv.writer(file)

            # Adding header row
            row = ['Timepoint','Amplitude','X-position','Sigma']
            if extra_columns is not None:
                for column in extra_columns:
                        row.append(column)
            writer.writerow(row)

            for peak in track.peaks.values():
                row = []
                row.append(peak.t)
                row.append(peak.a)
                row.append(peak.b)
                row.append(peak.c)

                if extra_columns is not None:
                    for column in extra_columns:
                        row.append(peak.measures[column])
                
                writer.writerow(row)

def write_intensity_traces(tracks, filepath):
    for track in tracks.values():
        with open(filepath+"_ID"+str(track.ID)+".csv", 'w', newline='') as file:
            writer = csv.writer(file)

            # Adding header row
            writer.writerow(['Timepoint','Intensity','Step'])

            for timepoint in track.intensity.keys():
                row = []
                row.append(timepoint)
                row.append(track.intensity.get(timepoint))
                try:
                    row.append(track.step_trace.get(timepoint))
                except:
                    None
                writer.writerow(row)

def create_track_overlay(tracks, image, show_labels=True):
    # Making 3 channels
    image = np.copy(image)
    image *= (255.0/image.max())
    image = np.expand_dims(image,axis=2)
    image_in = image
    image = np.append(image,image_in,axis=2)
    image = np.append(image,image_in,axis=2)
    
    for track in tracks.values():
        # Setting colour to this track with some transparency
        random.seed(track.ID)
        colour = hsv_to_rgb([random.random(),1,1])   

        # Drawing detected line
        for peak in track.peaks.values():
            image[math.floor(peak.b),math.floor(peak.t),0] = colour[0]*255
            image[math.floor(peak.b),math.floor(peak.t),1] = colour[1]*255
            image[math.floor(peak.b),math.floor(peak.t),2] = colour[2]*255
    
    img = Image.fromarray(image.astype(np.uint8))

    if show_labels:
        I1 = ImageDraw.Draw(img)
        myFont = ImageFont.truetype(pkg_resources.resource_filename('ctraptools','resources/fonts/Roboto-Regular.ttf'), 16)
                
        for track in tracks.values():
            random.seed(track.ID)
            colour = hsv_to_rgb([random.random(),1,1])   

            # Adding a label to the centre of each line       
            cent = list(track.peaks.values())[math.floor(len(track.peaks)/2)]
            I1.text((cent.t,cent.b), str(track.ID), font=myFont, fill = ((colour[0]*255).astype(np.uint8),(colour[1]*255).astype(np.uint8),(colour[2]*255).astype(np.uint8)))

    return img

def save_overlay(tracks, image, filepath, show_labels=True):
    img = create_track_overlay(tracks, image, show_labels=show_labels)

    img.save(filepath+"_IDs.png")

def save_plots(tracks, filepath):
    for track in tracks.values():
        fig = plt.figure(figsize=(8,6))
        plt.plot(list(track.intensity.keys()),list(track.intensity.values()))
        plt.plot(list(track.step_trace.keys()),list(track.step_trace.values()))
        plt.xlabel('Timepoint')
        plt.ylabel('Intensity')

        plt.savefig(filepath+"_ID"+str(track.ID)+".png")

def plot_gauss_for_frame(peaks, frame, image, half_t_w=3):
    plt.figure(figsize=(14,8))

    x, vals = get_raw_profile(image,frame,half_t_w)
    plt.plot(x,vals,color="black",linewidth=4)

    for peak in peaks.values():
        if peak.t == frame:
            g0 = gauss_1D(x,peak.a,peak.b,peak.c)
            plt.plot(x,g0)
    
    plt.xlabel('Position (px)')
    plt.ylabel('Intensity')
    plt.show()


