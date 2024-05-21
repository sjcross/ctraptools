from enum import Enum
from ctraptools.kymos.msd import MSD

import math
import numpy as np

class PeakMeasures(Enum):
    NN_DIST = 'Nearest neighbour distance'
    NN_ID = 'Nearest neighbour ID'
    INST_MSD = 'Instantaneous MSD'
    INST_D_COEFF = 'Instantaneous diffusion coefficient'

class TrackMeasures(Enum):
    MSD = 'MSD'

class Peak:
    def __init__(self, ID, t, a, b, c):
        self.ID = ID    # ID number
        self.t = t      # Timepoint
        self.a = a      # Amplitude
        self.b = b      # X-position
        self.c = c      # Sigma
        self.track = None # Assigned track
        self.measures = {}

    def calculate_nearest_neighbour(self, peaks, exclude_natching_ID=False):
        x_1 = self.b

        nn_dist = np.inf
        nn_id = 0

        for peak in peaks.values():
            # If comparing to same set of peaks from which this peak came, ignore if ID matches
            if exclude_natching_ID and peak.ID == self.ID:
                continue

            x_2 = peak.b

            dist = math.sqrt((x_2-x_1)*(x_2-x_1))

            if dist < nn_dist:
                nn_dist = dist
                nn_id = peak.ID

        self.measures[PeakMeasures.NN_DIST] = nn_dist
        self.measures[PeakMeasures.NN_ID] = nn_id

        return (nn_dist, nn_id)
    

class Track:
    def __init__(self, ID):
        self.ID = ID
        self.peaks = {}
        self.intensity = {}
        self.filtered = {}
        self.step_trace = {}
        self.steps = {}
        self.measures = {}
        
    def add_peak(self, peak):
        self.peaks[peak.t] = peak
        peak.track = self

    def measure_intensity(self, image, half_x_w=0, end_pad=100):
        for t in self.peaks.keys():
            peak = self.peaks.get(t)
            x = round(peak.b)
            self.intensity[t] = peak.a

        # Adding measurement points to the end
        t_start = max(max(self.peaks.keys())+1, 0)
        t_end = min(max(self.peaks.keys())+end_pad, image.shape[1])
        for t in range(t_start,t_end):
            self.intensity[t] = image[x-half_x_w:x+half_x_w,t].mean()
            
    def apply_temporal_filter(self,half_t_w=1):
        filtered = {}
        for t in self.intensity.keys():
            diff = abs(np.array(list(self.intensity.keys()))-t)
            filtered[t] = np.median(np.array(list(self.intensity.values()))[np.where(diff<=half_t_w)])

        self.intensity = filtered

    def get_timepoints(self):
        return [peak.t for peak in self.peaks.values()]
    
    def get_amplitudes(self):
        return [peak.a for peak in self.peaks.values()]

    def get_positions(self):
        return [peak.b for peak in self.peaks.values()]
    
    def get_sigmas(self):
        return [peak.c for peak in self.peaks.values()]
    
    def calculate_msd(self,spatial_scale=1,spatial_units="pixels",time_scale=1,time_units="frames",calculate_D=True,max_dt=50):
        curr_msd = MSD(self,spatial_scale=spatial_scale,spatial_units=spatial_units,time_scale=time_scale,time_units=time_units)

        if calculate_D:
            curr_msd.measure_diffusion_coefficient(max_dt=max_dt)

        self.measures[TrackMeasures.MSD] = curr_msd

    def calculate_instantaneous_msd(self, half_w=5, spatial_scale=1,spatial_units="pixels",time_scale=1,time_units="frames",calculate_D=True,max_dt=50):
        for peak in self.peaks.values():
            # Get subtrack
            subtrack = self.extract_subtrack(peak.t-half_w,peak.t+half_w)

            # Measure MSD
            curr_msd = MSD(subtrack,spatial_scale=spatial_scale,spatial_units=spatial_units,time_scale=time_scale,time_units=time_units)        

            if calculate_D:
                peak.measures[PeakMeasures.INST_D_COEFF] = curr_msd.measure_diffusion_coefficient(max_dt=max_dt)[0]

            peak.measures[PeakMeasures.INST_MSD] = curr_msd

    def calculate_stationary_probability(self, kymo_size, link_dist=3):
        counts = np.zeros(kymo_size)

        for curr_timepoint, curr_peak in sorted(list(self.peaks.items()), reverse=True):
            curr_position = curr_peak.b

            # Comparing to all other peaks in this track
            for prev_timepoint, prev_peak in sorted(list(self.peaks.items()), reverse=True):
                prev_position = prev_peak.b

                # Only process peaks from earlir timepoints
                if prev_timepoint >= curr_timepoint:
                    continue

                timepoint_separation = curr_timepoint - prev_timepoint
                position_separation = abs(curr_position-prev_position)

                if position_separation < link_dist:
                    pos_to_use = min(kymo_size[0]-1,max(0,round(curr_position)))
                    counts[pos_to_use,timepoint_separation] = counts[pos_to_use,timepoint_separation] + 1
                else:
                    # Once the peak has moved out of the linking range, stop recording history
                    break

        for f in range(kymo_size[1]):
            for x in range(kymo_size[0]):
                if counts[x,f] > 0:
                    counts[x,f] = counts[x,f] / (max(self.peaks.keys())-min(self.peaks.keys())-f-1)
                    # counts[x,f] = counts[x,f] / (image.shape[1]-f-1)

        return counts
    
    def extract_subtrack(self, start_t, end_t):
        subtrack = Track(str(self.ID)+"_"+str(start_t)+"-"+str(end_t))

        for peak in self.peaks.values():
            if peak.t >= start_t and peak.t <= end_t:
                subtrack.add_peak(peak)

        return subtrack

    