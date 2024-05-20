import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, SpanSelector
from scipy.signal import butter, sosfiltfilt

class KymoSelector:
    def __init__(self, kymo, px_min=0, px_max=255,channel=None):
        self._channel = channel
        self._abs_range_min = 0
        self._abs_range_max = kymo.get_image(channel='red').shape[1]-1 # Maximum number of frames minus 1
        self._range_min = self._abs_range_min
        self._range_max = self._abs_range_max
        self._px_min = px_min
        self._px_max = px_max
        self._min_slider = None # Sliders have to be retained to keep them active
        self._max_slider = None
        self._plot = None
        
        self._create_plot(kymo)

    def _create_plot(self, kymo):
        def on_span_select(range_min, range_max):
            self._range_min = max(self._abs_range_min,math.floor(range_min))
            self._range_max = min(self._abs_range_max,math.ceil(range_max))

        def update_min_intensity(val):
            self._px_min = val
            self._update_plot()
            
        def update_max_intensity(val):
            self._px_max = val
            self._update_plot()
        
        fig, ax1 = plt.subplots(1, figsize=(8,5))
        fig.subplots_adjust(left=0.25, bottom=0.35)
        
        if self._channel is None:
            raw_kymo = kymo.get_image()
        else:
            raw_kymo = kymo.get_image(channel=self._channel)
            
        self._plot = ax1.imshow(raw_kymo, aspect=3, cmap='viridis', norm=mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel('Distance ($\mu$m)', fontsize=12)
        
        self._span_selector_trace = SpanSelector(
            ax1,
            on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )

        ax_min_int = fig.add_axes([0.25, 0.2, 0.65, 0.03])
        self._min_slider = Slider(
            ax=ax_min_int,
            label='Min intensity',
            valmin=0,
            valmax=255, # Should be bit depth of image?
            valinit=self._px_min,
            valstep=1
        )
        self._min_slider.on_changed(update_min_intensity)

        ax_max_int = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self._max_slider = Slider(
            ax=ax_max_int,
            label='Max intensity',
            valmin=0,
            valmax=255, # Should be bit depth of image?
            valinit=self._px_max,
            valstep=1
        )
        self._max_slider.on_changed(update_max_intensity)    

    def _update_plot(self):
        self._plot.set_norm(mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        
    def get_display_range(self):
        return (self._px_min, self._px_max)
    
    def get_frame_range(self):
        return (self._range_min, self._range_max)
    
class TraceAnalyser:
    def __init__(self, kymo, time_ns, data, y_label = "Force (pN)", px_min=0, px_max=255, hist_n_bins=150):
        self._time_s = (time_ns - time_ns[0])/1000000000
        self._data = data
        self._y_label = y_label
        self._px_min = px_min
        self._px_max = px_max
        self._hist_n_bins = hist_n_bins

        self._fig = None
        self._crop_plot_ax = None
        self._filt_crop_plot = None

        self._span_selector_trace = None
        self._range_min = 0
        self._range_max = 1

        self._freq_slider = None
        self._order_slider = None
                
        self._create_plot(kymo)

    def _create_plot(self, kymo):
        def on_span_select(range_min, range_max):
            self._range_min = range_min
            self._range_max = range_max
            self._update_plot()
            
        # Initialising the plot
        self._fig, ((ax1, ax2), (ax3, ax4), (self._crop_plot_ax, self._hist_ax)) = plt.subplots(3, 2, figsize=(11, 11), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 1, 1.8]})
        self._fig.subplots_adjust(hspace=0.1, wspace=0.06)

        # Kymograph image
        kymo_line_length = kymo.size_um[0]
        raw_kymo = kymo.get_image(channel='red')
        t_stop = (kymo.stop-kymo.start)*1E-9
        ax1.imshow(raw_kymo, cmap='viridis', aspect='auto', extent=[0, t_stop, 0, kymo_line_length], norm=mpl.colors.Normalize(vmin=self._px_min, vmax=self._px_max, clip=False))
        ax1.set_ylabel('Distance ($\mu$m)')
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax1.set_xlabel("Time (s)")  
        ax1.xaxis.set_label_position('top') 

        ax2.axis('off')

        # Full length trace
        ax3.plot(self._time_s, self._data, color='black')
        ax3.set_ylabel(self._y_label)
        ax3.sharex(ax1)
        ax3.set_xticklabels([])
        self._span_selector_trace = SpanSelector(
            ax3,
            on_span_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )

        ax4.axis('off')

        # Cropped trace
        self._crop_plot_ax.set_xlabel("Time (s)")
        self._crop_plot_ax.set_ylabel(self._y_label)
        self._crop_plot_ax.plot(self._time_s, self._data, label='Raw data', color='black', alpha=.3)
        self._filt_crop_plot, = self._crop_plot_ax.plot([], [], label='Filtered data')

        # adjust the main plot to make room for the sliders
        self._fig.subplots_adjust(left=0.25, bottom=0.32)

        # Make a horizontal slider to set the filter frequency:
        self._freq_slider = Slider(
            ax=self._fig.add_axes([0.25, 0.2, 0.65, 0.03]),
            label='Butterworth filter frequency',
            valmin=0.1,
            valmax=1,
            valinit=0.4,
            valstep=0.05
        )
        
        # Make a horizontal slider to set the filter order:
        self._order_slider = Slider(
            ax=self._fig.add_axes([0.25, 0.17, 0.65, 0.03]),
            label='Butterworth filter order',
            valmin=1,
            valmax=20,
            valinit=1,
            valstep=1
        )

        # We can just call the standard _update_plot function
        def update_slider(val):
            self._update_plot()

        self._freq_slider.on_changed(update_slider)
        self._order_slider.on_changed(update_slider)

    def _update_plot(self):
        curr_time_s = self._time_s[(self._time_s > self._range_min) & (self._time_s < self._range_max)]
        curr_data = self._data[(self._time_s > self._range_min) & (self._time_s < self._range_max)]
        sos = butter(self._order_slider.val, self._freq_slider.val, 'lp', output='sos')
        filtered_data =  sosfiltfilt(sos, curr_data)
        
        counts, bins = np.histogram(curr_data, bins=self._hist_n_bins)
        
        region_min = self._data.min()
        region_max = self._data.max()
        region_range = region_max - region_min

        if len(self._time_s) >= 2:
            self._crop_plot_ax.set_xlim(self._range_min, self._range_max)
            self._crop_plot_ax.set_ylim(region_min - (region_range*0.05), region_max + (region_range*0.05))
            self._crop_plot_ax.legend(fontsize=9)
            self._filt_crop_plot.set_data(curr_time_s, filtered_data)
            
            self._hist_ax.cla()
            self._hist_ax.hist(bins[:-1], bins, weights=counts, orientation="horizontal")
            self._hist_ax.set_xlabel("Counts")
            self._hist_ax.set_title('Frequencies (raw force)', fontsize=8)
            self._hist_ax.sharey(self._crop_plot_ax)
            self._hist_ax.set_yticklabels([])
            self._hist_ax.minorticks_on()
            self._hist_ax.yaxis.set_tick_params(which='minor', bottom=False)

            self._fig.canvas.draw_idle()

def remove_tracks_by_id(tracks):
    ids_to_remove = input("Enter track IDs to remove: ").split(',')

    for id_to_remove in ids_to_remove:
        tracks.pop(id_to_remove)

def retain_tracks_by_id(tracks):
    ids_to_retain = input("Enter track IDs to retain: ").split(',')

    ids_to_remove = list(tracks.keys())

    for id_to_retain in ids_to_retain:
        ids_to_remove.remove(int(id_to_retain))

    for id_to_remove in ids_to_remove:
        tracks.pop(id_to_remove)
