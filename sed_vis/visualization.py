#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization
==================
This is module contains a simple visualizer to show event lists along with the audio.
The visualizer can show multiple event lists for the same reference audio allowing the
comparison of the reference and estimated event lists.

 .. image:: visualization.png


.. autosummary::
    :toctree: generated/

    EventListVisualizer
    EventListVisualizer.show

"""
from __future__ import print_function, absolute_import
from sed_vis.util import AudioPlayer, AudioThread
import dcase_util
import numpy
import math
import time

import scipy.fftpack
import scipy.signal
from numpy.lib.stride_tricks import as_strided
from sys import platform as _platform
import matplotlib
if _platform == "darwin":
    # MAC OS X
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm

from matplotlib.widgets import Button, SpanSelector
from matplotlib.patches import Rectangle


class EventListVisualizer(object):
    """Event List visualizer.

    Examples
    --------
    >>> # Load audio signal first
    >>> audio_container = dcase_util.containers.AudioContainer().load('data/audio.wav')
    >>> # Load event lists
    >>> reference_event_list = dcase_util.containers.MetaDataContainer().load('data/reference.txt')
    >>> estimated_event_list = dcase_util.containers.MetaDataContainer().load('data/estimated.txt')
    >>> event_lists = {'reference': reference_event_list, 'estimated': estimated_event_list}
    >>> # Visualize the data
    >>> vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,audio_signal=audio_container.data,sampling_rate=audio_container.fs)
    >>> vis.show()

    """

    def __init__(self, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        event_lists : dict of event lists
            Dict of event lists

        event_list_order : list
            Order of event list, if None alphabetical order used
            (Default value=None)

        active_events : list
            List of active sound event classes, if None all used.
            (Default value=None)

        audio_signal : np.ndarray
            Audio signal

        sampling_rate : int
            Sampling rate [0:96000]

        mode: str
            Signal visualization mode ['spectrogram', 'time_domain']
            (Default value = 'spectrogram')

        spec_hop_size : int
            Spectrogram calculation hop length in samples
            (Default value=256)

        spec_win_size: int ,
            Spectrogram calculation window length in samples
            (Default value=1024)

        spec_fft_size: int
            FFT length
            (Default value=1024)

        spec_cmap : str
            Color map used for spectrogram, see examples: http://matplotlib.org/examples/color/colormaps_reference.html
            (Default value='magma')

        spec_interpolation : str
            Matrix interpolation method for spectrogram (e.g. nearest, bilear, bicubic, quadric, gaussian)
            (Default value='nearest')

        event_roll_cmap : str
            Color map used for spectrogram, see examples: http://matplotlib.org/examples/color/colormaps_reference.html
            (Default value='rainbow')

        minimum_event_length : float > 0.0
            Minimum event length in seconds, shorten than given are filtered out from the output.
            (Default value=None)

        minimum_event_gap : float > 0.0
            Minimum allowed gap between events in seconds from same event label class.
            (Default value=None)

        color : color hex
            Main color code used in highlighting things

        use_blit : bool
            Use blit
            (Default value=False)

        publication_mode : bool
            Strip visual elements, can be used to prepare figures for publications.
            (Default value=False)

        show_selector : bool
            Show highlight selector
            (Default value=True)

        labels: dict
            Text labels overrides

        button_color : dict
            Button color overrides

        Returns
        -------
        Nothing

        """

        if kwargs.get('event_lists', []):
            self._event_lists = kwargs.get('event_lists', [])

            if kwargs.get('event_list_order') is None:
                self._event_list_order = sorted(self._event_lists.keys())
            else:
                self._event_list_order = kwargs.get('event_list_order')

            events = dcase_util.containers.MetaDataContainer()
            for event_list_label in self._event_lists:
                events += self._event_lists[event_list_label]

            self.event_labels = sorted(events.unique_event_labels, reverse=True)
            self.event_label_count = events.event_label_count

            if kwargs.get('active_events') is None:
                self.active_events = self.event_labels

            else:
                self.active_events = sorted(kwargs.get('active_events'), reverse=True)

            for name in self._event_lists:
                self._event_lists[name] = self._event_lists[name].process_events(
                    minimum_event_length=kwargs.get('minimum_event_length'),
                    minimum_event_gap=kwargs.get('minimum_event_gap')
                )

        else:
            self._event_lists = None

        if kwargs.get('audio_signal') is not None and kwargs.get('sampling_rate') is not None:
            audio_signal = kwargs.get('audio_signal') / numpy.max(numpy.abs(kwargs.get('audio_signal')))
            self.audio = AudioPlayer(
                signal=audio_signal,
                sampling_rate=kwargs.get('sampling_rate')
            )

        if kwargs.get('mode') not in ['spectrogram', 'time_domain']:
            self.mode = 'spectrogram'
        else:
            self.mode = kwargs.get('mode')

        self.auto_play = kwargs.get('auto_play', False)

        self.spec_hop_size = kwargs.get('spec_hop_size', 256)
        self.spec_win_size = kwargs.get('spec_win_size', 1024)
        self.spec_fft_size = kwargs.get('spec_fft_size', 1024)
        self.spec_cmap = kwargs.get('spec_cmap', 'magma')
        self.spec_interpolation =  kwargs.get('spec_interpolation', 'nearest')

        self.color = kwargs.get('color', '#339933')

        self.button_color = {
            'off': 'grey',
            'on': 'red'
        }

        self.button_color.update(kwargs.get('button_color',{}))
        self.labels = {
            'close': 'Close',
            'play': 'Play',
            'stop': 'Stop',
            'quit': 'Quit',
            'selection': 'Selection',
            'waveform': 'Waveform',
            'spectrogram': 'Spectrogram',
            'verification': 'Verification',
            'verification_info': '',
            'info': '',
        }

        self.labels.update(kwargs.get('labels',{}))

        self.indicator_line_color = self.color

        # Initialize members
        self.fig = None

        # Panels
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

        self.D = None
        self.x = None
        self.timedomain_locations = None

        self.begin_time = None
        self.end_time = None
        self.playback_offset = 0

        # Play indicators
        self.animation_event_roll_panel = None
        self.animation_selector_panel = None
        self.animation_highlight_panel = None

        self.event_panel_indicator_line = None
        self.selector_panel_indicator_line = None
        self.highlight_panel_indicator_line = None

        self.slider_time = None

        # Buttons
        self.buttons = {
            'play': True,
            'pause': False,
            'stop': True,
            'close': True,
            'quit': False,
            'verification': False,
        }
        self.buttons.update(kwargs.get('buttons',{}))

        self.button_play = None
        self.button_pause = None
        self.button_stop = None
        self.button_close = None

        self.use_blit = kwargs.get('use_blit', False)

        self.publication_mode = kwargs.get('publication_mode', False)
        self.show_selector = kwargs.get('show_selector', True)

        self.panel_title_font_size = 14
        self.legend_font_size = 12
        self.event_roll_label_font_size = 14
        self.event_roll_time_font_size = 10

        self.waveform_selector_point_hop = kwargs.get('waveform_selector_point_hop', 1000)
        self.waveform_highlight_point_hop = 100
        self.waveform_highlight_color = self.color

        self.selector_panel_height = 10
        self.highlight_panel_height = 25
        self.event_roll_panel_height = 50

        self.selector_panel_loc = 0
        self.highlight_panel_loc = 17
        self.event_roll_panel_loc = 45

        self.event_roll_item_opacity = 0.5
        self.fig_shape = (14, 6)

        self._quit = False

        if self.publication_mode:
            self.panel_title_font_size = 14
            self.legend_font_size = 16
            self.event_roll_time_font_size = 12

            self.spec_cmap = 'magma_r'
            self.spec_interpolation = 'bicubic'
            if not self.waveform_selector_point_hop:
                self.waveform_selector_point_hop = 5000
            self.waveform_highlight_point_hop = 500
            self.waveform_highlight_color = 'black'
            if self.show_selector:
                if self.mode == 'time_domain':
                    self.fig_shape = (30, 4)

                elif self.mode == 'spectrogram':
                    self.fig_shape = (20, 5)

                if self._event_lists:
                    if self.event_label_count == 1:
                        self.selector_panel_height = 10
                        self.highlight_panel_height = 33
                        self.event_roll_panel_height = 33

                        self.selector_panel_loc = 0
                        self.highlight_panel_loc = 17
                        self.event_roll_panel_loc = 53

                        self.event_roll_time_font_size = 16

                    else:
                        self.selector_panel_height = 10
                        self.highlight_panel_height = 15
                        self.event_roll_panel_height = 60

                        self.selector_panel_loc = 0
                        self.highlight_panel_loc = 17
                        self.event_roll_panel_loc = 35

                else:
                    self.selector_panel_height = 30
                    self.highlight_panel_height = 66
                    self.event_roll_panel_height = 0

                    self.selector_panel_loc = 0
                    self.highlight_panel_loc = 37
                    self.event_roll_panel_loc = 0

                    self.event_roll_time_font_size = 16

            else:
                if self.mode == 'time_domain':
                    self.fig_shape = (30, 4)

                elif self.mode == 'spectrogram':
                    self.fig_shape = (20, 4)

                self.selector_panel_height = 0
                self.highlight_panel_height = 15
                self.event_roll_panel_height = 75

                self.selector_panel_loc = 0
                self.highlight_panel_loc = 0
                self.event_roll_panel_loc = 17

            self.event_roll_item_opacity = 1.0

        self.label_colormap = cm.get_cmap(name=kwargs.get('event_roll_cmap','rainbow'))

    def generate_GUI(self):
        """Generates the visualizer GUI.

        Parameters
        ----------
        Nothing

        Returns
        -------
        Nothing

        """

        self.fig = plt.figure(figsize=self.fig_shape)

        # Selector panel
        # ====================================
        if self.show_selector:
            self.ax1 = plt.subplot2grid(shape=(100, 1), loc=(self.selector_panel_loc, 0), rowspan=self.selector_panel_height, colspan=1)

            self.timedomain_locations = numpy.arange(0, self.audio.signal.shape[0])

            self.ax1.fill_between(
                self.timedomain_locations[::self.waveform_selector_point_hop],
                self.audio.signal[::self.waveform_selector_point_hop],
                -self.audio.signal[::self.waveform_selector_point_hop],
                color='0.5'
            )

            plt.yticks([])
            plt.axis('tight')

            self.ax1.set_xlim(self.timedomain_locations[0], self.timedomain_locations[-1])
            self.ax1.set_ylim(-1, 1)

            self.time_ticks(
                locations=self.timedomain_locations,
                n_ticks=10,
                sampling_rate=self.audio.fs
            )

            self.ax1.yaxis.grid(False, which='major')
            self.ax1.yaxis.grid(False, which='minor')
            self.ax1.xaxis.grid(True, which='major')
            self.ax1.xaxis.grid(True, which='minor')
            self.ax1.yaxis.set_label_position("right")

            plt.title(self.labels['selection'], fontsize=self.panel_title_font_size)

        # Highlight panel
        # ====================================
        self.ax2 = plt.subplot2grid(shape=(100, 1), loc=(self.highlight_panel_loc, 0), rowspan=self.highlight_panel_height, colspan=1)
        self.x = numpy.arange(0, self.audio.duration_samples)

        self.begin_time = self.x[0] / float(self.audio.fs)
        self.end_time = self.x[-1] / float(self.audio.fs)

        if self.mode == 'spectrogram':
            self.D = self.get_spectrogram(
                audio=self.audio.signal,
                n_fft=self.spec_fft_size,
                win_length=self.spec_win_size,
                hop_length=self.spec_hop_size
            )

            self.plot_spectrogram(
                self.D,
                sampling_rate=self.audio.fs,
                interpolation=self.spec_interpolation,
                cmap=self.spec_cmap
            )

            if not self.publication_mode:
                self.ax2.yaxis.grid(False, which='major')
                self.ax2.yaxis.grid(False, which='minor')
                self.ax2.xaxis.grid(False, which='major')
                self.ax2.xaxis.grid(False, which='minor')

                plt.ylabel(self.labels['spectrogram'], fontsize=self.panel_title_font_size)
            else:
                self.ax2.get_yaxis().set_visible(False)

        elif self.mode == 'time_domain':

            self.ax2.fill_between(
                self.x[::self.waveform_highlight_point_hop], self.audio.signal[::self.waveform_highlight_point_hop], -self.audio.signal[::self.waveform_highlight_point_hop],
                color=self.waveform_highlight_color
            )

            self.ax2.set_ylim(-1, 1)
            self.ax2.set_xlim(self.x[0], self.x[-1])

            segment_begin = self.x[0] / float(self.audio.fs)
            segment_end = self.x[-1] / float(self.audio.fs)

            locs = numpy.arange(segment_begin, segment_end)
            plt.xlim([locs[0], locs[-1]])
            self.time_ticks(locations=locs, n_ticks=20)

            plt.yticks([])
            plt.xticks([])
            self.ax2.yaxis.grid(False, which='major')
            self.ax2.yaxis.grid(False, which='minor')

            self.ax2.xaxis.grid(True, which='major')
            self.ax2.xaxis.grid(True, which='minor')
            if not self.publication_mode:
                plt.ylabel(self.labels['waveform'], fontsize=self.panel_title_font_size)

            self.ax2.set_xlim(self.timedomain_locations[0], self.timedomain_locations[-1])

        self.ax2.yaxis.set_label_position("right")

        # Event roll panel
        # ====================================
        if self._event_lists:
            event_list_count = len(self._event_lists)

            self.begin_time = 0
            self.end_time = self.audio.duration_seconds

            if event_list_count == 1:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=self.event_label_count
                )
                self.ax3 = plt.subplot2grid(
                    shape=(100, 1),
                    loc=(self.event_roll_panel_loc, 0),
                    rowspan=self.event_roll_panel_height+10,
                    colspan=1
                )

            else:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=event_list_count
                )
                self.ax3 = plt.subplot2grid(
                    shape=(100, 1),
                    loc=(self.event_roll_panel_loc, 0),
                    rowspan=self.event_roll_panel_height,
                    colspan=1
                )

            m = cm.ScalarMappable(norm=norm, cmap=self.label_colormap)

            line_margin = 0.1
            y = 0
            annotation_height = (1.0-line_margin*2)/event_list_count

            for label in self.active_events:
                for event_list_id, event_list_label in enumerate(self._event_list_order):
                    offset = (len(self._event_list_order)-1-event_list_id) * annotation_height

                    event_y = y - 0.5 + line_margin + offset

                    # grid line
                    line = plt.Rectangle(
                        (0, y-0.5),
                        height=0.001,
                        width=self.end_time,
                        edgecolor='black',
                        facecolor='black'
                    )

                    plt.gca().add_patch(line)

                    for event in self._event_lists[event_list_label]:
                        if event['event_label'] == label:
                            event_length = event['offset'] - event['onset']

                            if 'probability' in event:
                                if event_list_count == 1:
                                    color = m.to_rgba(x=y + offset, alpha=event['probability'])
                                else:
                                    color = m.to_rgba(x=event_list_id, alpha=event['probability'])

                                rectangle = plt.Rectangle(
                                    (event['onset'], event_y),
                                    height=annotation_height,
                                    width=event_length,
                                    edgecolor='black',
                                    facecolor=color,
                                    linewidth=0,
                                    picker=5
                                )

                            else:
                                if event_list_count == 1:
                                    color = m.to_rgba(x=y + offset)
                                else:
                                    color = m.to_rgba(x=event_list_id)

                                rectangle = plt.Rectangle(
                                    (event['onset'], event_y),
                                    height=annotation_height,
                                    width=event_length,
                                    edgecolor='black',
                                    facecolor=color,
                                    linewidth=0,
                                    alpha=self.event_roll_item_opacity,
                                    picker=5
                                )


                            plt.gca().add_patch(rectangle)

                y += 1

            # grid line
            line = plt.Rectangle((0, y - 0.5),
                                 height=0.001,
                                 width=self.end_time,
                                 edgecolor='black',
                                 facecolor='black')
            plt.gca().add_patch(line)
            # Axis
            plt.axis([0, self.audio.duration_seconds, -0.5, len(self.active_events) + 0.5])
            locs = numpy.arange(0, self.audio.duration_seconds, 0.00001)

            plt.xlim([locs[0], locs[-1]])
            plt.axis('tight')

            # X axis
            self.ax3.xaxis.grid(True, which='major')
            self.ax3.xaxis.grid(True, which='minor')
            plt.tick_params(axis='x', which='major', labelsize=self.event_roll_time_font_size)

            # Y axis
            plt.yticks(
                numpy.arange(len(self.active_events)),
                self.active_events,
                fontsize=self.event_roll_label_font_size
            )

            plt.ylabel('Event Roll', fontsize=self.panel_title_font_size)
            self.ax3.yaxis.set_label_position('right')
            self.ax3.yaxis.grid(False, which='major')
            self.ax3.yaxis.grid(False, which='minor')

            # Set event list legends panel
            self.ax3.set_xlim(self.begin_time, self.end_time)
            if event_list_count > 1:
                span = 0
                for event_list_id, event_list_label in enumerate(self._event_list_order):

                    ax_legend_color = plt.axes([0.225+span, 0.02, 0.02, 0.02])
                    Button(
                        ax_legend_color,
                        '',
                        color=m.to_rgba(event_list_id),
                        hovercolor=m.to_rgba(event_list_id)
                    )

                    ax_legend_label = plt.axes([0.225+0.025+span, 0.02, 0.10, 0.04])
                    ax_legend_label.axis('off')
                    ax_legend_label.text(0, 0, event_list_label, fontsize=self.legend_font_size)
                    span += 0.15

        if self.show_selector:
            self.slider_time = SpanSelector(
                ax=self.ax1,
                onselect=self.on_select,
                minspan=None,
                direction='horizontal',
                span_stays=True,
                useblit=self.use_blit,
                onmove_callback=None,
                rectprops=dict(alpha=0.15, facecolor=self.color)
            )

        if not self.publication_mode:
            ax_legend_label = plt.axes([0.92, 0.02, 0.10, 0.04])
            ax_legend_label.axis('off')
            ax_legend_label.text(0, 0, 'sed_vis', fontsize=16)

            # Buttons
            # ====================================
            ax_play = plt.axes([0.125, 0.93, 0.07, 0.04])
            ax_stop = plt.axes([0.205, 0.93, 0.07, 0.04])
            ax_close = plt.axes([0.92, 0.93, 0.07, 0.04])

            self.button_play = Button(
                ax_play,
                self.labels['play'],
                color=self.button_color['off'],
                hovercolor=self.button_color['on']
            )

            self.button_stop = Button(
                ax_stop,
                self.labels['stop'],
                color=self.button_color['off'],
                hovercolor=self.button_color['on']
            )

            self.button_close = Button(
                ax_close,
                self.labels['close'],
                color=self.button_color['off'],
                hovercolor=self.button_color['on']
            )

            self.button_play.on_clicked(self.on_play)
            self.button_stop.on_clicked(self.on_stop)
            self.button_close.on_clicked(self.on_close_window)

            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        else:
            plt.subplots_adjust(left=0.12, bottom=0.05, right=.97, top=0.95, wspace=0, hspace=0)

        if self.auto_play:
            self.on_play(None)

    def show(self):
        """Shows the visualizer.

        Parameters
        ----------
        Nothing

        Returns
        -------
        Nothing

        """
        self.generate_GUI()
        plt.show()

    def save(self, filename=None):
        if filename:
            self.generate_GUI()
            plt.savefig(filename, bbox_inches='tight')

    def on_close_window(self, event):
        if self.audio.playing:
            self.audio.stop()
            self.audio = None

        plt.close(self.fig)

    def on_pick(self, event):
        if isinstance(event.artist, Rectangle):
            if self.audio.playing:
                self.audio.stop()  # Stop current playback
                try:
                    self.event_panel_indicator_line.set_visible(False)
                    self.event_panel_indicator_line.remove()
                except:
                    pass

                if self.animation_event_roll_panel is not None:
                    self.animation_event_roll_panel._stop()
                    self.animation_event_roll_panel = None

                try:
                    self.selector_panel_indicator_line.set_visible(False)
                    self.selector_panel_indicator_line.remove()
                except:
                    pass

                if self.animation_selector_panel is not None:
                    self.animation_selector_panel._stop()
                    self.animation_selector_panel = None

                try:
                    self.highlight_panel_indicator_line.set_visible(False)
                    self.highlight_panel_indicator_line.remove()
                except:
                    pass

                if self.animation_highlight_panel is not None:
                    self.animation_highlight_panel._stop()
                    self.animation_highlight_panel = None

                self.fig.canvas.draw()

                time.sleep(0.25)  # Wait until playback has stopped

            self.playback_offset = event.artist.get_x()
            self.audio.play(
                offset=event.artist.get_x(),
                duration=event.artist.get_width()
            )

            # Set up play indicators animations
            self.animation_event_roll_panel = animation.FuncAnimation(
                self.fig,
                self.event_roll_panel_play_indicator_update,
                init_func=self.event_roll_panel_play_indicator_init,
                interval=10,
                blit=self.use_blit,
                repeat=False
            )

            self.animation_selector_panel = animation.FuncAnimation(
                self.fig,
                self.selector_panel_play_indicator_update,
                init_func=self.selector_panel_play_indicator_init,
                interval=10,
                blit=self.use_blit,
                repeat=False
            )

            self.animation_highlight_panel = animation.FuncAnimation(
                self.fig,
                self.highlight_panel_play_indicator_update,
                init_func=self.highlight_panel_play_indicator_init,
                interval=10,
                blit=self.use_blit,
                repeat=False
            )

            self.fig.canvas.draw()

    def on_select(self, x_min, x_max):
        x_min = int(x_min)
        x_max = int(x_max)
        if math.fabs(x_min-x_max) < 10:
            # Reset highlight
            self.begin_time = self.x[0] / float(self.audio.fs)
            self.end_time = self.x[-1] / float(self.audio.fs)
            if self.ax3:
                self.ax3.set_xlim(self.begin_time, self.end_time)

            # Set signal highlight panel
            if self.mode == 'spectrogram':
                self.ax2.set_xlim(0, self.D.shape[1])
            elif self.mode == 'time_domain':
                self.ax2.set_xlim(self.timedomain_locations[0], self.timedomain_locations[-1])

            self.slider_time.stay_rect.set_visible(False)

        else:
            # Set annotation panel
            self.begin_time = float(x_min) / self.audio.fs
            self.end_time = float(x_max) / self.audio.fs
            if self.ax3:
                self.ax3.set_xlim(self.begin_time, self.end_time)

            # Set signal highlight panel
            if self.mode == 'spectrogram':
                spec_min = int(x_min / float(self.spec_hop_size))
                spec_max = int(x_max / float(self.spec_hop_size))

                self.ax2.set_xlim(spec_min, spec_max)

            elif self.mode == 'time_domain':
                index_min, index_max = numpy.searchsorted(self.x, (x_min, x_max))
                index_max = min(len(self.x) - 1, index_max)
                this_x = self.timedomain_locations[index_min:index_max]
                self.ax2.set_xlim(this_x[0], this_x[-1])

            self.slider_time.stay_rect.set_visible(True)

        self.fig.canvas.draw()

    def on_play(self, event):
        if self.audio.playing:
            self.audio.stop()  # Stop current playback
            try:
                self.event_panel_indicator_line.set_visible(False)
                self.event_panel_indicator_line.remove()
            except:
                pass

            if self.animation_event_roll_panel is not None:
                self.animation_event_roll_panel._stop()
                self.animation_event_roll_panel = None

            try:
                self.selector_panel_indicator_line.set_visible(False)
                self.selector_panel_indicator_line.remove()
            except:
                pass

            if self.animation_selector_panel is not None:
                self.animation_selector_panel._stop()
                self.animation_selector_panel = None

            try:
                self.highlight_panel_indicator_line.set_visible(False)
                self.highlight_panel_indicator_line.remove()
            except:
                pass

            if self.animation_highlight_panel is not None:
                self.animation_highlight_panel._stop()
                self.animation_highlight_panel = None

            self.fig.canvas.draw()

            time.sleep(0.25)  # Wait until playback has stopped

        self.audio.play(
            offset=self.begin_time,
            duration=self.end_time-self.begin_time
        )

        self.button_play.color = self.button_color['on']
        self.button_play.hovercolor = self.button_color['on']

        self.button_stop.color = self.button_color['off']
        self.button_stop.hovercolor = self.button_color['off']

        self.playback_offset = self.begin_time

        self.animation_event_roll_panel = animation.FuncAnimation(
            self.fig,
            self.event_roll_panel_play_indicator_update,
            init_func=self.event_roll_panel_play_indicator_init,
            interval=50,
            blit=self.use_blit,
            repeat=False
        )

        self.animation_selector_panel = animation.FuncAnimation(
            self.fig,
            self.selector_panel_play_indicator_update,
            init_func=self.selector_panel_play_indicator_init,
            interval=50,
            blit=self.use_blit,
            repeat=False
        )

        self.animation_highlight_panel = animation.FuncAnimation(
            self.fig,
            self.highlight_panel_play_indicator_update,
            init_func=self.highlight_panel_play_indicator_init,
            interval=50,
            blit=self.use_blit,
            repeat=False
        )

        self.fig.canvas.draw()

    def on_pause(self, event):
        self.audio.pause()

        self.button_play.color = self.button_color['off']
        self.button_play.hovercolor = self.button_color['off']

        self.button_stop.color = self.button_color['off']
        self.button_stop.hovercolor = self.button_color['off']

        self.fig.canvas.draw()

    def on_stop(self, event):
        self.audio.stop()

        self.button_play.color = self.button_color['off']
        self.button_play.hovercolor = self.button_color['off']

        self.button_stop.color = self.button_color['off']
        self.button_stop.hovercolor = self.button_color['off']

        self.fig.canvas.draw()

    def on_quit(self, event):
        self._quit = True
        self.on_close_window(event=event)

    @property
    def quit(self):
        return self._quit

    def event_roll_panel_play_indicator_init(self):
        indicator_width = (self.end_time-self.begin_time) / 1000
        if indicator_width > 0.5:
            indicator_width = 0.5

        self.event_panel_indicator_line = patches.Rectangle(
            (self.playback_offset + self.audio.get_time(), -0.5),
            height=self.event_label_count,
            width=indicator_width,
            edgecolor=self.indicator_line_color,
            facecolor=self.indicator_line_color,
            alpha=0.8
        )

        self.ax3.add_patch(self.event_panel_indicator_line)
        return self.event_panel_indicator_line,

    def event_roll_panel_play_indicator_update(self, i):
        if self.audio.playing:
            self.event_panel_indicator_line.set_x(self.playback_offset + self.audio.get_time())
        else:
            self.event_panel_indicator_line.set_visible(False)
            if self.animation_event_roll_panel is not None:
                self.animation_event_roll_panel.event_source.stop()
                self.animation_event_roll_panel = None

            self.fig.canvas.draw()

        return self.event_panel_indicator_line,

    def selector_panel_play_indicator_init(self):
        self.selector_panel_indicator_line = patches.Rectangle(
            (0, -1),
            height=2,
            width=0.5,
            edgecolor=self.indicator_line_color,
            facecolor=self.indicator_line_color,
            alpha=0.8
        )

        self.ax1.add_patch(self.selector_panel_indicator_line)
        return self.selector_panel_indicator_line,

    def selector_panel_play_indicator_update(self, i):
        if self.audio.playing:
            self.selector_panel_indicator_line.set_x((self.playback_offset + self.audio.get_time())*self.audio.fs)
        else:
            self.selector_panel_indicator_line.set_visible(False)
            if self.animation_selector_panel is not None:
                self.animation_selector_panel.event_source.stop()
                self.animation_selector_panel = None

        return self.selector_panel_indicator_line,

    def highlight_panel_play_indicator_init(self):
        indicator_width = 0.5
        if self.mode == 'spectrogram':
            indicator_height = self.spec_fft_size
            indicator_y = 0

        elif self.mode == 'time_domain':
            indicator_height = 2
            indicator_y = -1

        else:
            indicator_height = 2
            indicator_y = -1

        self.highlight_panel_indicator_line = patches.Rectangle(
            (0, indicator_y),
            height=indicator_height,
            width=indicator_width,
            edgecolor=self.indicator_line_color,
            facecolor=self.indicator_line_color,
            alpha=0.8
        )

        self.ax2.add_patch(self.highlight_panel_indicator_line)
        return self.highlight_panel_indicator_line,

    def highlight_panel_play_indicator_update(self, i):
        if self.audio.playing:
            if self.mode == 'spectrogram':
                self.highlight_panel_indicator_line.set_x(
                    (self.playback_offset + self.audio.get_time()) * self.audio.fs / float(self.spec_hop_size)
                )

            elif self.mode == 'time_domain':
                self.highlight_panel_indicator_line.set_x(
                    (self.playback_offset + self.audio.get_time()) * self.audio.fs
                )

        else:
            self.highlight_panel_indicator_line.set_visible(False)
            if self.animation_highlight_panel is not None:
                self.animation_highlight_panel.event_source.stop()
                self.animation_highlight_panel = None

        return self.highlight_panel_indicator_line,

    def time_ticks(self, locations, n_ticks=10, sampling_rate=44100):
        times = self.samples_to_time(locations, sampling_rate=sampling_rate)
        positions = numpy.linspace(0, len(locations)-1, n_ticks, endpoint=True).astype(int)
        locations = locations[positions]
        times = times[positions]
        times = ['{:0.2f}s'.format(t) for t in times]

        return plt.xticks(locations, times)

    @staticmethod
    def time_to_samples(time, sampling_rate=44100):
        return (numpy.atleast_1d(time) * sampling_rate).astype(int)

    @staticmethod
    def samples_to_time(samples, sampling_rate=44100):
        return numpy.atleast_1d(samples) / float(sampling_rate)

    @staticmethod
    def get_spectrogram(audio, n_fft=256, win_length=1024, hop_length=1024):
        fft_window = scipy.signal.hann(win_length, sym=False).reshape((-1, 1))

        audio = numpy.pad(array=audio,
                          pad_width=int(n_fft // 2),
                          mode='reflect')

        n_frames = 1 + int((len(audio) - n_fft) / hop_length)
        y_frames = as_strided(x=audio,
                              shape=(n_fft, n_frames),
                              strides=(audio.itemsize, int(hop_length * audio.itemsize)))

        S = numpy.empty((int(1 + n_fft // 2), y_frames.shape[1]), dtype=numpy.complex64, order='F')

        max_memory_block = 2**8 * 2**10
        n_columns = int(max_memory_block / (S.shape[0] * S.itemsize))

        for bl_s in range(0, S.shape[1], n_columns):
            bl_t = min(bl_s + n_columns, S.shape[1])

            # RFFT and Conjugate here to match phase from DPWE code
            S[:, bl_s:bl_t] = scipy.fftpack.fft(fft_window * y_frames[:, bl_s:bl_t], axis=0)[:S.shape[0]].conj()

        magnitude = numpy.abs(S) ** 2

        ref = numpy.max(magnitude)
        amin=1e-10
        top_db = 80.0

        log_spec = 10.0 * numpy.log10(numpy.maximum(amin, magnitude))
        log_spec -= 10.0 * numpy.log10(numpy.maximum(amin, ref))

        log_spec = numpy.maximum(log_spec, log_spec.max() - top_db)

        return log_spec

    @staticmethod
    def plot_spectrogram(data, sampling_rate=44100, n_yticks=5, interpolation='nearest', cmap='magma'):

        axes = plt.imshow(data, aspect='auto', origin='lower', interpolation=interpolation, cmap=plt.get_cmap(cmap))

        # X axis
        plt.xticks([])

        # Y axis
        positions = numpy.linspace(0, data.shape[0]-1, n_yticks, endpoint=True).astype(int)
        values = numpy.linspace(0, 0.5 * sampling_rate, data.shape[0], endpoint=True).astype(int)

        t_log = (data.shape[0] * (1 - numpy.logspace(-numpy.log2(data.shape[0]), 0, data.shape[0], base=2, endpoint=True))[::-1]).astype(int)
        t_inv = numpy.arange(len(t_log))
        for i in range(len(t_log)-1):
            t_inv[t_log[i]:t_log[i+1]] = i

        plt.yticks(positions, values[t_inv[positions]])

        return axes


class EventListVerifier(EventListVisualizer):
    def __init__(self, *args, **kwargs):
        super(EventListVerifier, self).__init__(*args, **kwargs)
        self.verification_answer_id = None
        self.verification_answer_value = None
        self.verification_values = kwargs.get('verification_values', [
            'A',
            'B',
            'C'
        ])

        self.verification_button_colors = kwargs.get('verification_button_colors',[
            {
                'off': '#AAAAAA',
                'on':'#b32400',
            },
            {
                'off': '#AAAAAA',
                'on': '#996600',

            },
            {
                'off': '#AAAAAA',
                'on': '#009933',
            },
        ])

        self.button_color = kwargs.get('button_color', {
            'off': '#AAAAAA',
            'on': '#666666'
        })

        self.button_verification = {}
        self.button_verification_axis = {}

        if self.buttons['verification']:
            # Verification panel
            # ====================================
            self.selector_panel_loc = 20
            self.highlight_panel_loc = 37
            self.event_roll_panel_loc = 75

            self.selector_panel_height = 10
            self.highlight_panel_height = 35
            self.event_roll_panel_height = 20

    def generate_GUI(self):
        """Generates the visualizer GUI."""

        self.fig = plt.figure(figsize=self.fig_shape)

        # Selector panel
        # ====================================
        if self.show_selector:
            self.ax1 = plt.subplot2grid(
                shape=(100, 1),
                loc=(self.selector_panel_loc, 0),
                rowspan=self.selector_panel_height,
                colspan=1
            )

            self.timedomain_locations = numpy.arange(0, self.audio.signal.shape[0])

            self.ax1.fill_between(
                self.timedomain_locations[::self.waveform_selector_point_hop],
                self.audio.signal[::self.waveform_selector_point_hop],
                -self.audio.signal[::self.waveform_selector_point_hop],
                color='0.5'
            )

            plt.yticks([])
            plt.axis('tight')

            self.ax1.set_xlim(self.timedomain_locations[0], self.timedomain_locations[-1])
            self.ax1.set_ylim(-1, 1)

            self.time_ticks(
                locations=self.timedomain_locations,
                n_ticks=10,
                sampling_rate=self.audio.fs
            )

            self.ax1.yaxis.grid(False, which='major')
            self.ax1.yaxis.grid(False, which='minor')
            self.ax1.xaxis.grid(True, which='major')
            self.ax1.xaxis.grid(True, which='minor')
            self.ax1.yaxis.set_label_position("right")

            plt.title(self.labels['selection'], fontsize=self.panel_title_font_size)

        # Highlight panel
        # ====================================
        self.ax2 = plt.subplot2grid(
            shape=(100, 1),
            loc=(self.highlight_panel_loc, 0),
            rowspan=self.highlight_panel_height,
            colspan=1
        )

        self.x = numpy.arange(0, self.audio.duration_samples)

        self.begin_time = float(self.x[0]) / self.audio.fs
        self.end_time = float(self.x[-1]) / self.audio.fs

        if self.mode == 'spectrogram':
            self.D = self.get_spectrogram(
                audio=self.audio.signal,
                n_fft=self.spec_fft_size,
                win_length=self.spec_win_size,
                hop_length=self.spec_hop_size
            )

            self.plot_spectrogram(
                self.D,
                sampling_rate=self.audio.fs,
                interpolation=self.spec_interpolation,
                cmap=self.spec_cmap
            )

            if not self.publication_mode:
                self.ax2.yaxis.grid(False, which='major')
                self.ax2.yaxis.grid(False, which='minor')
                self.ax2.xaxis.grid(False, which='major')
                self.ax2.xaxis.grid(False, which='minor')

                plt.ylabel(self.labels['spectrogram'], fontsize=self.panel_title_font_size)
            else:
                self.ax2.get_yaxis().set_visible(False)

        elif self.mode == 'time_domain':

            self.ax2.fill_between(
                self.x[::self.waveform_highlight_point_hop],
                self.audio.signal[::self.waveform_highlight_point_hop],
                -self.audio.signal[::self.waveform_highlight_point_hop],
                color=self.waveform_highlight_color
            )

            self.ax2.set_ylim(-1, 1)
            self.ax2.set_xlim(self.x[0], self.x[-1])

            segment_begin = self.time_to_samples(time=0, sampling_rate=self.audio.fs)
            segment_end = self.time_to_samples(time=self.audio.duration_seconds, sampling_rate=self.audio.fs)
            locs = numpy.arange(segment_begin, segment_end)
            plt.xlim([locs[0], locs[-1]])
            self.time_ticks(locations=locs, n_ticks=20)

            plt.yticks([])
            plt.xticks([])
            self.ax2.yaxis.grid(False, which='major')
            self.ax2.yaxis.grid(False, which='minor')

            self.ax2.xaxis.grid(True, which='major')
            self.ax2.xaxis.grid(True, which='minor')

            if not self.publication_mode:
                plt.ylabel(self.labels['waveform'], fontsize=self.panel_title_font_size)

        self.ax2.yaxis.set_label_position("right")

        plt.axis('tight')

        # Event roll panel
        # ====================================
        if self._event_lists:
            event_list_count = len(self._event_lists)

            self.begin_time = 0
            self.end_time = self.audio.duration_seconds

            if event_list_count == 1:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=self.event_label_count
                )
                self.ax3 = plt.subplot2grid(
                    shape=(100, 1),
                    loc=(self.event_roll_panel_loc, 0),
                    rowspan=self.event_roll_panel_height+10,
                    colspan=1
                )

            else:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=event_list_count
                )
                self.ax3 = plt.subplot2grid(
                    shape=(100, 1),
                    loc=(self.event_roll_panel_loc, 0),
                    rowspan=self.event_roll_panel_height,
                    colspan=1
                )

            m = cm.ScalarMappable(
                norm=norm,
                cmap=self.label_colormap
            )

            line_margin = 0.1
            y = 0
            annotation_height = (1.0-line_margin*2)/event_list_count

            for label in self.active_events:
                for event_list_id, event_list_label in enumerate(self._event_list_order):
                    offset = (len(self._event_list_order)-1-event_list_id) * annotation_height

                    event_y = y - 0.5 + line_margin + offset

                    # grid line
                    line = plt.Rectangle(
                        (0, y-0.5),
                        height=0.001,
                        width=self.end_time,
                        edgecolor='black',
                        facecolor='black'
                    )

                    plt.gca().add_patch(line)

                    for event in self._event_lists[event_list_label]:
                        if event['event_label'] == label:
                            event_length = event['offset'] - event['onset']

                            if event_list_count == 1:
                                color = m.to_rgba(y + offset)
                            else:
                                color = m.to_rgba(event_list_id)

                            rectangle = plt.Rectangle(
                                (event['onset'], event_y),
                                height=annotation_height,
                                width=event_length,
                                edgecolor='black',
                                facecolor=color,
                                linewidth=0,
                                alpha=self.event_roll_item_opacity,
                                picker=5
                            )

                            plt.gca().add_patch(rectangle)
                y += 1

            # Grid line
            line = plt.Rectangle(
                (0, y - 0.5),
                height=0.001,
                width=self.end_time,
                edgecolor='black',
                facecolor='black'
            )

            plt.gca().add_patch(line)

            # Axis
            plt.axis([0, self.audio.duration_seconds, -0.5, len(self.active_events) + 0.5])
            locs = numpy.arange(0, self.audio.duration_seconds, 0.00001)

            plt.xlim([locs[0], locs[-1]])
            plt.axis('tight')

            # X axis
            self.ax3.xaxis.grid(True, which='major')
            self.ax3.xaxis.grid(True, which='minor')
            plt.tick_params(
                axis='x',
                which='major',
                labelsize=self.event_roll_time_font_size
            )

            # Y axis
            plt.yticks(
                numpy.arange(len(self.active_events)),
                self.active_events,
                fontsize=self.event_roll_label_font_size
            )

            plt.ylabel('Event Roll', fontsize=self.panel_title_font_size)
            self.ax3.yaxis.set_label_position('right')
            self.ax3.yaxis.grid(False, which='major')
            self.ax3.yaxis.grid(False, which='minor')

            # Set event list legends panel
            self.ax3.set_xlim(self.begin_time, self.end_time)
            if event_list_count > 1:
                span = 0
                for event_list_id, event_list_label in enumerate(self._event_list_order):

                    ax_legend_color = plt.axes([0.225+span, 0.02, 0.02, 0.02])
                    Button(
                        ax_legend_color,
                        '',
                        color=m.to_rgba(event_list_id),
                        hovercolor=m.to_rgba(event_list_id)
                    )

                    ax_legend_label = plt.axes([0.225+0.025+span, 0.02, 0.10, 0.04])
                    ax_legend_label.axis('off')
                    ax_legend_label.text(0, 0, event_list_label, fontsize=self.legend_font_size)
                    span += 0.15

        if self.show_selector:
            self.slider_time = SpanSelector(
                ax=self.ax1,
                onselect=self.on_select,
                minspan=None,
                direction='horizontal',
                span_stays=True,
                useblit=self.use_blit,
                onmove_callback=None,
                rectprops=dict(alpha=0.15, facecolor=self.color)
            )

        if not self.publication_mode:
            ax_legend_label = plt.axes([0.92, 0.02, 0.10, 0.04])
            ax_legend_label.axis('off')
            ax_legend_label.text(0, 0, 'sed_vis', fontsize=16)

            # Buttons
            # ====================================
            if self.buttons['play']:
                ax_play = plt.axes([0.125, 0.93, 0.07, 0.04])
                self.button_play = Button(
                    ax_play,
                    self.labels['play'],
                    color=self.button_color['off'],
                    hovercolor=self.button_color['on']
                )

                self.button_play.on_clicked(self.on_play)

            if self.buttons['stop']:
                ax_stop = plt.axes([0.205, 0.93, 0.07, 0.04])
                self.button_stop = Button(
                    ax_stop,
                    self.labels['stop'],
                    color=self.button_color['off'],
                    hovercolor=self.button_color['on']
                )

                self.button_stop.on_clicked(self.on_stop)

            if self.buttons['close']:
                ax_close = plt.axes([0.92, 0.93, 0.07, 0.04])
                self.button_close = Button(
                    ax_close,
                    self.labels['close'],
                    color=self.button_color['off'],
                    hovercolor=self.button_color['on']
                )

                self.button_close.on_clicked(self.on_close_window)

            if self.buttons['quit']:
                ax_quit = plt.axes([0.78, 0.93, 0.07, 0.04])
                self.button_quit = Button(
                    ax_quit,
                    self.labels['quit'],
                    color=self.button_color['off'],
                    hovercolor=self.button_color['on']
                )

                self.button_quit.on_clicked(self.on_quit)

            if self.buttons['verification']:
                start_x = 0.125 + 0.2 #25
                end_x = 0.125+0.775 -0.2 #1 - start_x
                width = (end_x - start_x)
                spacing = width / float(len(self.verification_values))

                # Verification text
                plt.axes([0.125, 0.88, 0.775, 0.04], frameon=False)
                plt.xticks([])
                plt.yticks([])
                plt.text(
                    0.5, 0.5,
                    self.labels['verification'],
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center'
                )

                # Verification-info text
                plt.axes([0.00, 0.80, 0.125 + 0.18, 0.08], frameon=False)
                plt.xticks([])
                plt.yticks([])
                plt.text(
                    1, 0.5,
                    self.labels['verification_info'],
                    fontsize=18,
                    horizontalalignment='right',
                    verticalalignment='center'
                )

                for i, label in enumerate(self.verification_values):
                    self.button_verification_axis[i] = plt.axes([start_x+(i*spacing), 0.80, 0.12, 0.08])
                    self.button_verification[i] = Button(
                        self.button_verification_axis[i],
                        label,
                        color=self.verification_button_colors[i]['off'],
                        hovercolor=self.verification_button_colors[i]['off']
                    )

                    self.button_verification[i].label.set_fontsize(12)
                    self.button_verification[i].on_clicked(self.on_verification)

            # Info text
            plt.axes([start_x, 0.02, width, 0.04], frameon=False)
            plt.xticks([])
            plt.yticks([])
            plt.text(0.5, 0.5,
                     self.labels['info'],
                     fontsize=12,
                     horizontalalignment='center',
                     verticalalignment='center'
                     )

            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        else:
            plt.subplots_adjust(
                left=0.12,
                bottom=0.05,
                right=.97,
                top=0.95,
                wspace=0,
                hspace=0
            )

        if self.auto_play:
            self.on_play(None)

    def on_verification(self, event):
        for i in self.button_verification:
            if event.inaxes == self.button_verification_axis[i]:
                self.button_verification[i].color = self.verification_button_colors[i]['on']
                self.button_verification[i].hovercolor = self.verification_button_colors[i]['on']
                self.verification_answer_id =i
                self.verification_answer_value = self.verification_values[i]

            else:
                self.button_verification[i].color = self.verification_button_colors[i]['off']
                self.button_verification[i].hovercolor = self.verification_button_colors[i]['off']

    def get_answer(self):
        return self.verification_answer_id, self.verification_answer_value

