#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video generation
==================
This is module contains a simple video generator to show events along with the video.

.. autosummary::
    :toctree: generated/

    VideoGenerator
    VideoGenerator.generate

"""
from __future__ import print_function, absolute_import
import dcase_util
import numpy
import math
import time
import textwrap
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors

class VideoGenerator(object):
    """Video generator

    Examples
    --------

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

        Returns
        -------
        Nothing

        """

        import cv2

        self.ui = dcase_util.ui.FancyPrinter()

        if kwargs.get('event_lists', []):
            self._event_lists = kwargs.get('event_lists', [])

            if kwargs.get('event_list_order') is None:
                self._event_list_order = sorted(self._event_lists.keys())
            else:
                self._event_list_order = kwargs.get('event_list_order')

            events = dcase_util.containers.MetaDataContainer()
            for event_list_label in self._event_lists:
                events += self._event_lists[event_list_label]

            self.event_labels = sorted(events.unique_event_labels, reverse=False)
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

        from IPython import embed
        #embed()

        self.source = kwargs.get('source', None)
        self.target_video = kwargs.get('target', None)

        self.target_video_final = kwargs.get('target_video', None)

        self.title = kwargs.get('title', None)
        self.intro_text = kwargs.get('intro_text', None)

        self.video_data = cv2.VideoCapture(filename=self.source)

        self.audio_data = dcase_util.containers.AudioContainer().load(filename=self.source, mono=True, fs=16000)
        self.audio_info = dcase_util.utils.get_audio_info(filename=self.source)
        self.fs = self.audio_data.fs

        self.frame_width = kwargs.get('frame_width', 1280)
        self.frame_height = kwargs.get('frame_height', 720)

        if self.video_data.isOpened():
            self.fps = self.video_data.get(cv2.CAP_PROP_FPS)
            self.frame_duration = 1.0 / self.fps
            self.frame_count = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps
            self.audio_only = False
        else:
            self.fps = 29.97
            self.frame_duration = 1.0 / self.fps
            self.duration = self.audio_data.duration_sec
            self.frame_count = int(self.duration / self.frame_duration)
            self.audio_only = True

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        self.output = cv2.VideoWriter(self.target_video, fourcc,  self.fps, (self.frame_width, self.frame_height))

        self.logos = kwargs.get('logos', {})
        self.panels = {
            'header': {
                'enable': False,
                'header': None,
                'size': {
                    'height': 50,
                    'width': self.frame_width,
                },
                'top_y': 0,
                'top_x': 0,
                'color': (240, 240, 240)
            },
            'footer': {
                'enable': False,
                'header': None,
                'size': {
                    'height': 50,
                    'width': self.frame_width,
                },
                'top_y': self.frame_height-50,
                'top_x': 0,
                'color': (240, 240, 240)
            },
            'video': {
                'enable': False,
                'header': None,
                'size': {
                  'width': 480,
                  'height': 300
                },
                'top_y': 30,
                'top_x': 10,
                'color': (0, 0, 0),
                'frame_color': (100,100,100)
            },
            'spectrogram': {
                'enable': False,
                'header': None,
                'size': {
                    'width': 762,
                    'height': 300,
                },
                'sub_panel': {
                    'right': {
                        'width': 70
                    }
                },
                #'right_sub_panel_width': 70,
                'top_y': 30,
                'top_x': None,
                'color': (0, 0, 0),
                'frame_color': (100,100,100)

            },
            'mid_header': {
                'enable': False,
                'header': None,
                'size': {
                    'width': self.frame_width,
                    'height': 15,
                },
                'top_y': 368,
                'top_x': 0,
                'color': (255, 255, 255),
                'frame_color': None
            },
            'event_list': {
                'enable': False,
                'header': None,
                'size': {
                    'width': 0,
                    'height': 0,
                },
                'top_y': 370,
                'top_x': 0,
                'color': (255, 255, 255),
                'frame_color': None
            },
            'event_roll': {
                'enable': False,
                'header': None,
                'size': {
                    'width': 762,
                    'height': 0,
                },
                'sub_panel': {
                    'right': {
                        'width': 70
                    },
                    'bottom': {
                        'height': 20
                    }
                },
                #'right_sub_panel_width': 70,
                'top_y': 370,
                'top_x': 0,
                'color': (155, 155, 155),
                'frame_color': (100,100,100)
            },
            'event_text': {
                'enable': False,
                'header': None,
                'size': {
                    'width': 0,
                    'height': 0,
                },
                'top_y': 370,
                'top_x': 0,
                'color': (255, 255, 255),
                'frame_color': None
            },
            'video_dummy': {
                'enable': False,
                'header': None,
                'size': {
                    'width': None,
                    'height': None
                },
            }
        }
        self.panels['video_dummy']['size']['width'] = self.panels['video']['size']['width']
        self.text = kwargs.get('text', {
            'header': None,
            'footer': None,
            'video': 'Video stream',
            'spectrogram': 'Spectrogram',
            'mid_header': {
                'header': 'Sound Event Detection',
                'description': 'Description of the application'
            },
            'event_list': 'Events',
            'event_roll': 'Events',
            'event_text': 'Events',
            'video_dummy': None
        })

        for panel_id in self.panels:
            if panel_id in self.text and self.text[panel_id]:
                if 'header' not in self.text[panel_id]:
                    self.panels[panel_id]['header'] = self.text[panel_id]
                else:
                    self.panels[panel_id]['header'] = self.text[panel_id]['header']

                if 'description' in self.text[panel_id]:
                    self.panels[panel_id]['description'] = self.text[panel_id]['description']

            else:
                self.panels[panel_id]['header'] = None

        self.layout = kwargs.get('layout',
            [
                #['header'],
                #['video'],
                ['video', 'spectrogram'],
                #['mid_header'],
                #['video_dummy', 'event_roll'],
                ['event_list', 'event_roll'],
                #['event_text'],
                ['footer']
            ]
        )

        self.margin = {
            'window':{
                'left': 8,
                'right': 8,
            },
            'layout': {
                'column': 20,
                'row': 20
            }
        }

        self.ui.row('row', 'col', 'panel', 'x', 'y', 'height', 'width',  indent=4)
        self.ui.row_sep()

        align_tmp = []
        for row_id, row in enumerate(self.layout):
            if 'event_roll' in row:
                align_tmp.append(row.index('event_roll'))
            if 'event_text' in row:
                align_tmp.append(row.index('event_text'))
            if 'spectrogram' in row:
                align_tmp.append(row.index('spectrogram'))

        # Align spectrogram and event_roll
        if len(set(align_tmp)) == 1:
            for row_id, row in enumerate(self.layout):
                for col_id, col in enumerate(row):
                    if col == 'spectrogram':
                        self.panels['event_roll']['top_x'] = self.panels[col]['top_x']
                        self.panels['event_roll']['size']['width'] = self.panels[col]['size']['width']
                    elif col == 'event_roll':
                        self.panels['spectrogram']['top_x'] = self.panels[col]['top_x']
                        self.panels['spectrogram']['size']['width'] = self.panels[col]['size']['width']
                    elif col == 'event_text':
                        self.panels['spectrogram']['top_x'] = self.panels[col]['top_x']
                        self.panels['spectrogram']['size']['width'] = self.panels[col]['size']['width']

        # Get row height sum
        row_height_sum = 0
        for row_id, row in enumerate(self.layout):
            row_heights = []
            for col_id, col in enumerate(row):
                row_heights.append(self.panels[col]['size']['height'])
            row_height_sum += max(row_heights) + self.margin['layout']['row']

        current_y = 0
        for row_id, row in enumerate(self.layout):
            current_x = self.margin['window']['left']
            row_heights = []
            row_widths = []
            for col_id, col in enumerate(row):
                if self.panels[col]['size']['height'] is not None:
                    row_heights.append(self.panels[col]['size']['height'])
                if self.panels[col]['size']['width'] is not None:
                    row_widths.append(self.panels[col]['size']['width'])

            current_row_height = max(row_heights)
            row_width_sum = sum(row_widths)

            if current_row_height == 0:
                current_row_height = self.frame_height - row_height_sum

            for col_id, col in enumerate(row):
                self.panels[col]['enable'] = True
                if col != 'footer' and col != 'header':
                    self.panels[col]['top_x'] = current_x
                    self.panels[col]['top_y'] = current_y
                    self.panels[col]['size']['height'] = current_row_height
                    if self.panels[col]['size']['width'] == 0:
                        self.panels[col]['size']['width'] = self.frame_width - self.margin['window']['left'] - row_width_sum - self.margin['layout']['column'] - self.margin['window']['right']

                    if len(row) == 1:
                        if col != 'video':
                            self.panels[col]['size']['width'] = self.frame_width - self.panels[col]['top_x'] - self.margin['window']['right']

                    current_x += self.panels[col]['size']['width'] + self.margin['layout']['column']

                    if self.panels[col]['header'] is not None:
                        self.panels[col]['active_panel'] = {
                            'x': self.panels[col]['top_x'],
                            'y': self.panels[col]['top_y'] + 28,
                            'size': {
                                'width': self.panels[col]['size']['width'] ,
                                'height': self.panels[col]['size']['height'] - 28,
                            }
                        }

                    else:
                        self.panels[col]['active_panel'] = {
                            'x': self.panels[col]['top_x'],
                            'y': self.panels[col]['top_y'],
                            'size': {
                                'width': self.panels[col]['size']['width'] ,
                                'height': self.panels[col]['size']['height'],
                            }
                        }

                    if 'sub_panel' in self.panels[col] and 'right' in self.panels[col]['sub_panel'] and self.panels[col]['sub_panel']['right']['width']:
                        self.panels[col]['active_panel']['size']['width'] = self.panels[col]['active_panel']['size']['width'] - self.panels[col]['sub_panel']['right']['width']
                    if 'sub_panel' in self.panels[col] and 'bottom' in self.panels[col]['sub_panel'] and self.panels[col]['sub_panel']['bottom']['height']:
                        self.panels[col]['active_panel']['size']['height'] = self.panels[col]['active_panel']['size']['height'] - self.panels[col]['sub_panel']['bottom']['height']

                self.ui.row(row_id, col_id, col, self.panels[col]['top_x'], self.panels[col]['top_y'], self.panels[col]['size']['height'], self.panels[col]['size']['width'] )

            current_y += current_row_height + self.margin['layout']['row']
            self.ui.row_sep()

        if self.panels['footer']['enable'] and self.logos:
            for logo_id in self.logos:
                self.logos[logo_id] = cv2.imread(self.logos[logo_id], cv2.IMREAD_UNCHANGED)
                if self.logos[logo_id].shape[2] == 3:
                    self.logos[logo_id] = cv2.cvtColor( self.logos[logo_id], cv2.COLOR_RGB2RGBA)
                    self.logos[logo_id][:, :, 3] = 1

                if 'header' in logo_id:
                    self.logos[logo_id] = self.image_resize(
                        image_data=self.logos[logo_id],
                        height=self.panels['header']['size']['height'] - 6
                    )
                elif 'footer' in logo_id:
                    self.logos[logo_id] = self.image_resize(
                        image_data=self.logos[logo_id],
                        height=self.panels['footer']['size']['height'] - 6
                    )

        self.header_color = (100, 100, 100)
        self.label_colormap = cm.get_cmap(name=kwargs.get('event_roll_cmap', 'rainbow'))

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.spectrogram_move = 10
        self.spectrogram_segment_duration = (self.panels['spectrogram']['size']['width']-6) / self.spectrogram_move * self.frame_duration
        self.cmap = matplotlib.cm.get_cmap('magma')
        self.norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)

        self.fft_size = 1024

        self.ui.section_header('Video generator')
        self.ui.data('Source', self.source)
        self.ui.data('Target', self.target_video)
        self.ui.sep()
        self.ui.line('Audio')
        self.ui.data('fs', self.audio_info['fs'], 'hz', indent=4)
        self.ui.sep()
        self.ui.line('Frame')
        self.ui.data('Width', self.frame_width, 'px', indent=4)
        self.ui.data('Height', self.frame_height, 'px', indent=4)
        self.ui.data('Count', self.frame_count, indent=4)
        self.ui.data('Per second', self.fps, indent=4)
        self.ui.sep()
        self.ui.line('Video')
        self.ui.data('Duration', self.duration, 'sec', indent=4)
        self.ui.data('Duration [MIN:SEC]', '{min:d}:{sec:0.1f}'.format(min=int(self.duration/60), sec=self.duration%60), indent=4)

    def move_image(self, image_data, x, y):
        import cv2
        return cv2.warpAffine(image_data, numpy.float32([[1, 0, x], [0, 1, y]]), (image_data.shape[1], image_data.shape[0]))

    def image_resize(self, image_data, width=None, height=None):
        import cv2
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image_data.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image_data

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image_data, dim, interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def place_image(target_image, source_image, x=0, y=0):
        target_image[y:y + source_image.shape[0],x:x + source_image.shape[1]] = source_image

    @staticmethod
    def place_image_watermark(target_image, source_image, x=0, y=0, opacity=100):
        import cv2
        opacity = opacity / 100
        temp_image = target_image.copy()

        def overlay_image(source_image, overlay, position=(0, 0), scale=1):
            import cv2
            overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
            h, w, _ = overlay.shape
            rows, cols, _ = source_image.shape
            y_, x_ = position[0], position[1]

            for i in range(h):
                for j in range(w):
                    if x_ + i >= rows or y_ + j >= cols:
                        continue
                    alpha = float(overlay[i][j][3] / 255.0)
                    source_image[x_ + i][y_ + j] = alpha * overlay[i][j][:3] + (1 - alpha) * source_image[x_ + i][y_ + j]

            return source_image

        overlay = overlay_image(
            source_image=temp_image,
            overlay=source_image,
            position=(x,y)
        )

        output = target_image.copy()

        cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
        return output

    def transparentOverlay(self, src, overlay, pos=(0, 0), scale=1):
        """
        :param src: Input Color Background Image
        :param overlay: transparent Image (BGRA)
        :param pos:  position where the image to be blit.
        :param scale : scale factor of transparent image.
        :return: Resultant Image
        """
        import cv2
        #overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  # Size of foreground
        rows, cols, _ = src.shape  # Size of background Image
        y, x = pos[0], pos[1]  # Position of foreground/overlay image
        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
        return src

    def addImageWatermark(self, logo, frame, opacity=100, pos=(0, 0)):
        import cv2
        opacity = opacity / 100
        tempImg = frame.copy()

        overlay = self.transparentOverlay(tempImg, logo, pos)
        output = frame.copy()

        cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)
        return output

    @staticmethod
    def place_image_transparent(target_image, source_image, x=None, y=None):
        # Function based on https://stackoverflow.com/a/71701023
        import numpy
        target_h, target_w, target_channels = target_image.shape
        source_h, source_w, source_channels = source_image.shape

        assert target_channels == 3, f'target_image image should have exactly 3 channels (RGB). found:{target_channels}'
        assert source_channels == 4, f'source_image image should have exactly 4 channels (RGBA). found:{source_channels}'

        # center by default
        if x is None: x = (target_w - source_w) // 2
        if y is None: y = (target_h - source_h) // 2

        w = min(source_w, target_w, source_w + x, target_w - x)
        h = min(source_h, target_h, source_h + y, target_h - y)

        if w < 1 or h < 1: return

        # clip foreground and background images to the overlapping regions
        target_x = max(0, x)
        target_y = max(0, y)
        source_x = max(0, x * -1)
        source_y = max(0, y * -1)
        source_image = source_image[source_y:source_y + h, source_x:source_x + w]
        target_subsection = target_image[target_y:target_y + h, target_x:target_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = source_image[:, :, :3]
        alpha_channel = source_image[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = numpy.dstack((alpha_channel, alpha_channel, alpha_channel))

        # combine the background with the overlay image weighted by alpha
        composite = target_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        target_image[target_y:target_y + h, target_x:target_x + w] = composite

    def generate(self):
        import cv2
        import librosa
        from IPython import embed

        #embed()
        frame_index = 0
        frame_time = 0

        event_set_font_scale = 0.6

        # Create spectrogram
        self.audio_data.normalize()
        if self.panels['spectrogram']['enable']:
            self.spectrogram = numpy.zeros((
                self.panels['spectrogram']['active_panel']['size']['height']-3*2,
                self.panels['spectrogram']['active_panel']['size']['width']-3*2,
                3
            ), numpy.uint8)

            self.spectrogram.fill(0)

            S = numpy.abs(librosa.stft(
                y=self.audio_data.data,
                n_fft=self.fft_size,
                win_length=882,  # int(self.fft_size/4),
                hop_length=int(self.frame_duration*self.audio_data.fs),  # 441, # int(self.fft_size/4))
                center=True
            ))
            S_mag = librosa.amplitude_to_db(S, ref=1.0)
            video_spectrogram = numpy.zeros((S_mag.shape[0], S_mag.shape[1], 3), numpy.uint8)

            for i in range(0, S_mag.shape[0]):
                current_pixels = self.cmap(self.norm(S_mag[i, :]), bytes=True)
                video_spectrogram[i, :, :] = current_pixels[:, 0:3]

            video_spectrogram = numpy.flip(video_spectrogram)

            video_spectrogram = cv2.resize(
                src=video_spectrogram,
                dsize=(self.frame_count*self.spectrogram_move, self.spectrogram.shape[0]),
                interpolation=cv2.INTER_AREA
            )

            video_spectrogram_offset = int(self.spectrogram.shape[1]/2)

            self.spectrogram[:, video_spectrogram_offset:, :] = video_spectrogram[:, 0:video_spectrogram_offset, :]

        event_list_colors = {}
        if self.panels['event_list']['enable'] or self.panels['event_roll']['enable'] or self.panels['event_text']['enable']:
            line_margin = 0.1
            annotation_height = (1.0 - line_margin * 2) / len(self._event_lists)

            if len(self._event_lists) == 1:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=self.event_label_count
                )
            else:
                norm = colors.Normalize(
                    vmin=0,
                    vmax=len(self._event_lists)
                )
            m = cm.ScalarMappable(
                norm=norm,
                cmap=self.label_colormap
            )
            for event_list_id, event_list_label in enumerate(self._event_list_order):
                if event_list_label not in event_list_colors:
                    offset = (len(self._event_list_order) - 1 - event_list_id) * annotation_height

                    if len(self._event_lists) == 1:
                        color = m.to_rgba(offset)
                    else:
                        color = m.to_rgba(event_list_id)
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    event_list_colors[event_list_label] = color

        if self.panels['event_roll']['enable']:
            self.event_roll_move = 10
            self.event_roll_segment_duration = (self.panels['event_roll']['active_panel']['size']['width'] - 6) / self.event_roll_move * self.frame_duration
            self.event_roll = numpy.zeros((
                self.panels['event_roll']['active_panel']['size']['height'] - 6,
                self.panels['event_roll']['active_panel']['size']['width'] - 6, 3),
                numpy.uint8
            )

            self.event_roll.fill(0)


            #(self.frame_count * self.event_roll_move) / self.audio_data.duration_sec
            #embed()
            time_axis_multiplier = 1000
            video_event_roll = numpy.zeros((self.panels['event_roll']['active_panel']['size']['height']-6, (self.frame_count * self.event_roll_move), 3), numpy.uint8)
            video_event_roll.fill(255)

            for grid_x in range(0, (self.frame_count * self.event_roll_move), int(1.0*(self.frame_count * self.event_roll_move) / self.audio_data.duration_sec)):
                for dot in range(0, self.panels['event_roll']['active_panel']['size']['height']-6, 5):
                    cv2.line(
                        img=video_event_roll,
                        pt1=(grid_x, dot),
                        pt2=(grid_x, dot+1),
                        color=(150, 150, 150),
                        thickness=1
                    )

            line_margin = 0.1
            y = 0
            event_list_count = len(self._event_lists)
            annotation_height = (1.0-line_margin*2)/event_list_count

            label_lane_height = int((video_event_roll.shape[0]) / self.event_label_count)

            sublane_margin = 3
            sublane_height = int(label_lane_height / len(self._event_list_order)-sublane_margin)

            for label in self.active_events:
                for event_list_id, event_list_label in enumerate(self._event_list_order):
                    offset = (len(self._event_list_order)-1-event_list_id) * annotation_height

                    event_y1 = y + (event_list_id * (sublane_height+sublane_margin))
                    event_y2 = y + (event_list_id * (sublane_height+sublane_margin)) + sublane_height

                    for event in self._event_lists[event_list_label]:
                        if event['event_label'] == label:
                            video_event_roll = cv2.rectangle(
                                img=video_event_roll,
                                pt1=(int(event['onset']*(self.frame_count * self.event_roll_move) / self.audio_data.duration_sec), event_y1),
                                pt2=(int(event['offset']*(self.frame_count * self.event_roll_move) / self.audio_data.duration_sec), event_y2),
                                color=event_list_colors[event_list_label],
                                thickness=-1
                            )

                y += label_lane_height

            #video_event_roll = cv2.resize(
            #    src=video_event_roll,
            #    dsize=(self.frame_count*self.event_roll_move, self.panels['event_roll']['size']['height'] - 6),
            #    interpolation=cv2.INTER_NEAREST #INTER_AREA
            #)

            video_event_roll_offset = int(self.event_roll.shape[1] / 2)
            self.event_roll[:, video_event_roll_offset:, :] = video_event_roll[:, 0:video_event_roll_offset, :]


        while True:
            current_output_frame = numpy.zeros((self.frame_height, self.frame_width, 3), numpy.uint8)
            current_output_frame.fill(255)
            if not self.audio_only:
                video_ret, video_frame = self.video_data.read()
                if self.panels['video']['enable'] and video_ret:
                    video_frame = self.image_resize(
                        image_data=video_frame,
                        height=self.panels['video']['active_panel']['size']['height']-(3*2),
                        width=self.panels['video']['active_panel']['size']['width']-(3*2),
                    )
            else:
                if frame_time <  self.duration:
                   video_ret = True
                else:
                   video_ret = False

            # Layout
            if self.panels['header']['enable']:
                # Header
                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['header']['top_x'], self.panels['header']['top_y'] - 1),
                    pt2=(self.panels['header']['top_x'] + self.panels['header']['size']['width'], self.panels['header']['top_y'] + self.panels['header']['size']['height']),
                    color=self.panels['header']['color'],
                    thickness=-1
                )
                if self.panels['header']['header']:
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['header']['header'], self.font, 0.8, 2)
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['header']['header'],
                        org=(self.panels['header']['top_x'] + int(self.panels['header']['size']['width'] / 2) - int(text_width/2), self.panels['header']['top_y'] + int(self.panels['header']['size']['height'] / 2) + text_height),
                        fontFace=self.font,
                        fontScale=0.8,
                        color=self.header_color,
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                if 'header_left' in self.logos:
                    self.place_image_transparent(
                        target_image=current_output_frame,
                        source_image=self.logos['header_left'],
                        x=self.panels['header']['top_x']+4,
                        y=self.panels['header']['top_y']+3
                    )

                if 'header_right' in self.logos:
                    self.place_image_transparent(
                        target_image=current_output_frame,
                        source_image=self.logos['header_right'],
                        x=self.panels['header']['top_x']+self.panels['header']['size']['width']-self.logos['header_right'].shape[1]-4,
                        y=self.panels['header']['top_y']+3
                    )

            if self.panels['video']['enable']:
                # Video box
                if self.panels['video']['header']:
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['video']['header'], self.font, 0.8, 1)
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['video']['header'],
                        org=(self.panels['video']['top_x'], self.panels['video']['top_y'] + text_height),
                        fontFace=self.font,
                        fontScale=0.8,
                        color=self.header_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['video']['active_panel']['x'], self.panels['video']['active_panel']['y']),
                    pt2=(self.panels['video']['active_panel']['x'] + self.panels['video']['active_panel']['size']['width'], self.panels['video']['active_panel']['y'] + self.panels['video']['active_panel']['size']['height']),
                    color = self.panels['video']['color'],
                    thickness=-1
                )
                if self.panels['video']['frame_color']:
                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['video']['active_panel']['x'], self.panels['video']['active_panel']['y']),
                        pt2=(self.panels['video']['active_panel']['x'] + self.panels['video']['active_panel']['size']['width'],
                             self.panels['video']['active_panel']['y'] + self.panels['video']['active_panel']['size']['height']),
                        color=self.panels['video']['frame_color'],
                        thickness=3
                    )

                if video_ret:
                    # Place video frame
                    offset_y = int(((self.panels['video']['active_panel']['size']['height']) - video_frame.shape[0])/2)
                    offset_x = int((self.panels['video']['active_panel']['size']['width'] - video_frame.shape[1])/2)
                    self.place_image(
                        target_image=current_output_frame,
                        source_image=video_frame,
                        x=offset_x + self.panels['video']['active_panel']['x'],
                        y=offset_y + self.panels['video']['active_panel']['y']
                    )

            if self.panels['spectrogram']['enable']:
                if self.panels['spectrogram']['header']:
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['spectrogram']['header'], self.font, 0.8, 1)
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['spectrogram']['header'],
                        org=(self.panels['spectrogram']['top_x'], self.panels['spectrogram']['top_y'] + text_height),
                        fontFace=self.font,
                        fontScale=0.8,
                        color=self.header_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                self.spectrogram = self.move_image(image_data=self.spectrogram, x=-self.spectrogram_move, y=0)

                spectrogram_slice_start = video_spectrogram_offset + frame_index * self.spectrogram_move
                spectrogram_slice_end = video_spectrogram_offset + (frame_index + 1) * self.spectrogram_move

                if spectrogram_slice_start >= video_spectrogram.shape[1] or spectrogram_slice_end >= video_spectrogram.shape[1]:
                    spectrogram_slice = numpy.zeros((video_spectrogram.shape[0], self.spectrogram_move, 3), numpy.uint8)
                else:
                    spectrogram_slice = video_spectrogram[:, spectrogram_slice_start:spectrogram_slice_end, :]

                if spectrogram_slice.shape[1] > 0:
                    self.spectrogram[:, -(self.spectrogram_move + 1):-1, :] = spectrogram_slice[:, :, :]

                # Spectrogram box
                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['spectrogram']['active_panel']['x'], self.panels['spectrogram']['active_panel']['y']),
                    pt2=(self.panels['spectrogram']['active_panel']['x'] + self.panels['spectrogram']['active_panel']['size']['width'],
                         self.panels['spectrogram']['active_panel']['y'] + self.panels['spectrogram']['active_panel']['size']['height']),
                    color=self.panels['spectrogram']['color'],
                    thickness=-1
                )
                if self.panels['spectrogram']['frame_color']:
                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['spectrogram']['active_panel']['x'], self.panels['spectrogram']['active_panel']['y']),
                        pt2=(self.panels['spectrogram']['active_panel']['x'] + self.panels['spectrogram']['active_panel']['size']['width'],
                             self.panels['spectrogram']['active_panel']['y'] + self.panels['spectrogram']['active_panel']['size']['height']),
                        color=self.panels['spectrogram']['frame_color'],
                        thickness=3
                    )


                # Place spectrogram
                offset_y = int((self.panels['spectrogram']['active_panel']['size']['height'] - self.spectrogram.shape[0]) / 2)
                offset_x = int((self.panels['spectrogram']['active_panel']['size']['width'] - self.spectrogram.shape[1]) / 2)
                self.place_image(
                    target_image=current_output_frame,
                    source_image=self.spectrogram,
                    x=offset_x + self.panels['spectrogram']['active_panel']['x'],
                    y=offset_y + self.panels['spectrogram']['active_panel']['y']
                )

                freq_index = 0
                tick_hz = self.fs / 4.0
                for y in range(self.panels['spectrogram']['active_panel']['y'], self.panels['spectrogram']['active_panel']['y'] + self.spectrogram.shape[0] + 1 , int(self.spectrogram.shape[0] / 4)):
                    cv2.putText(
                        img=current_output_frame,
                        text='{index:.0f}kHz'.format(index=(self.fs - freq_index * tick_hz) / 1000),
                        org=(self.panels['spectrogram']['active_panel']['x'] + self.spectrogram.shape[1] + 10, y + 4),
                        fontFace=self.font,
                        fontScale=0.5,
                        color=(100, 100, 100),
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )
                    freq_index += 1
                if 0:
                    cv2.putText(
                        img=current_output_frame,
                        text='Frequency',
                        org=(self.panels['spectrogram']['active_panel']['x'] + self.spectrogram.shape[1] + 55, self.panels['spectrogram']['active_panel']['y'] + 4 + int(self.spectrogram.shape[0] / 2)),
                        fontFace=self.font,
                        fontScale=0.5,
                        color=(100, 100, 100),
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                shade = numpy.zeros_like(current_output_frame, numpy.uint8)

                cv2.rectangle(
                    img=shade,
                    pt1=(
                        self.panels['spectrogram']['active_panel']['x'] + int(self.panels['spectrogram']['active_panel']['size']['width'] / 2),
                        self.panels['spectrogram']['active_panel']['y'] + 3
                    ),
                    pt2=(
                        self.panels['spectrogram']['active_panel']['x'] + self.panels['spectrogram']['active_panel']['size']['width'] - 3,
                        self.panels['spectrogram']['active_panel']['y'] + self.spectrogram.shape[0] + 1
                    ),
                    color=(255, 255, 255),
                    thickness=cv2.FILLED
                )
                mask = shade.astype(bool)
                current_output_frame_shaded = current_output_frame.copy()
                alpha = 0.5
                current_output_frame_shaded[mask] = cv2.addWeighted(current_output_frame, alpha, shade, 1 - alpha, 0)[mask]
                current_output_frame = current_output_frame_shaded

            if self.panels['mid_header']['enable']:
                current_panel_text_y = self.panels['mid_header']['top_y'] + 10
                text_margin = 10
                if self.panels['mid_header']['header']:
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['mid_header']['header'],
                        org=(self.panels['mid_header']['top_x'], current_panel_text_y),
                        fontFace=self.font,
                        fontScale=1,
                        color=(100, 100, 100),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['mid_header']['header'], self.font, 1, 2)

                if self.panels['mid_header']['description']:
                    (text_width2, text_height2), text_baseline = cv2.getTextSize(self.panels['mid_header']['description'], self.font, 0.5, 1)
                    if self.panels['mid_header']['description']:
                        cv2.putText(
                            img=current_output_frame,
                            text=self.panels['mid_header']['description'],
                            org=(self.frame_width-text_width2 - text_margin, current_panel_text_y),
                            fontFace=self.font,
                            fontScale=0.5,
                            color=(180, 180, 180),
                            thickness=1,
                            lineType=cv2.LINE_AA
                        )

                    current_panel_text_y += text_height + 10

            if self.panels['event_list']['enable']:
                current_panel_text_y = self.panels['event_list']['top_y']

                # Labels
                column_margin = 20
                column_width = 60
                event_label_font_scale = 0.5
                text_margin = 5

                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['event_list']['top_x'], self.panels['event_list']['top_y']),
                    pt2=(self.panels['event_list']['top_x'] + self.panels['event_list']['size']['width'], self.panels['event_list']['top_y'] + self.panels['event_list']['size']['height']),
                    color=self.panels['event_list']['color'],
                    thickness=-1
                )
                if self.panels['event_list']['frame_color']:
                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['event_list']['top_x'], self.panels['event_list']['top_y']),
                        pt2=(self.panels['event_list']['top_x'] + self.panels['event_list']['size']['width'],
                             self.panels['event_list']['top_y'] + self.panels['event_list']['size']['height']),
                        color=self.panels['event_list']['frame_color'],
                        thickness=3
                    )

                (text_width, text_height), text_baseline = cv2.getTextSize('text', self.font, event_set_font_scale, 2)
                event_list_height = self.panels['event_list']['size']['height'] - 15 - text_baseline

                for scale in reversed(range(0, 100, 1)):
                    current_height = 0
                    current_width = 0
                    for col_id, event_list_label in enumerate(self._event_lists):
                        column_max_width = 0
                        for event_label_id, event_label in enumerate(self.event_labels):
                            (text_width, text_height), text_baseline = cv2.getTextSize(event_label, fontFace=self.font, fontScale=scale / 10, thickness=2)
                            if col_id == 0:
                                current_height += text_height + int(text_height / 2)
                            if text_width > column_max_width:
                                column_max_width = text_width
                        current_width += column_max_width

                    if (current_height <= event_list_height) and (current_width <= self.panels['event_list']['size']['width']):
                        event_label_font_scale = scale / 10.0
                        break

                row_margin = int(text_height / 2)

                for col_id, event_list_label in enumerate(self._event_lists):
                    (text_width, text_height), text_baseline = cv2.getTextSize(text=event_list_label, fontFace=self.font, fontScale=event_set_font_scale, thickness=2)
                    if column_width < text_width:
                        column_width = text_width
                    for event_label_id, event_label in enumerate(self.event_labels):
                        (text_width, text_height), text_baseline = cv2.getTextSize(text=event_label, fontFace=self.font, fontScale=event_label_font_scale, thickness=1)
                        if column_width < text_width:
                            column_width = text_width

                for col_id, event_list_label in enumerate(self._event_lists):
                    current_text_y = current_panel_text_y + 15

                    current_active_events = self._event_lists[event_list_label].filter_time_segment(
                        start=frame_time,
                        stop=frame_time + self.frame_duration
                    ).unique_event_labels

                    # Event group title
                    (text_width, text_height), text_baseline = cv2.getTextSize(event_list_label, self.font, event_set_font_scale, 2)

                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['event_list']['top_x'] + col_id*(column_width + column_margin),
                             current_text_y-text_height-5),
                        pt2=(text_margin*2 + self.panels['event_list']['top_x'] + col_id*(column_width + column_margin)-2,
                             current_text_y+5),
                        color=event_list_colors[event_list_label],
                        thickness=-1
                    )
                    cv2.line(
                        img=current_output_frame,
                        pt1=(self.panels['event_list']['top_x'], current_text_y + int(text_baseline)+4),
                        pt2=(self.panels['event_list']['top_x'] + self.panels['event_list']['size']['width'],
                             current_text_y + int(text_baseline) + 4),
                        color=(200, 200, 200),
                        thickness=1
                    )

                    cv2.putText(
                        img=current_output_frame,
                        text=event_list_label,
                        org=(text_margin*2 + self.panels['event_list']['top_x'] + col_id*(column_width + column_margin),
                             current_text_y),
                        fontFace=self.font,
                        fontScale=event_set_font_scale,
                        color=(100, 100, 100),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                    current_text_y += text_baseline
                    (text_width, text_height), text_baseline = cv2.getTextSize(event_list_label, self.font, event_set_font_scale, 2)
                    current_text_y += text_baseline + row_margin * 3

                    for event_label_id, event_label in enumerate(self.event_labels):
                        if event_label in current_active_events:
                            cv2.putText(
                                img=current_output_frame,
                                text=event_label,
                                org=(text_margin*2 + self.panels['event_list']['top_x'] + col_id * (column_width + column_margin),
                                     current_text_y),
                                fontFace=self.font,
                                fontScale=event_label_font_scale,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA
                            )
                        else:
                            cv2.putText(
                                img=current_output_frame,
                                text=event_label,
                                org=(text_margin*2 + self.panels['event_list']['top_x'] + col_id * (column_width + column_margin),
                                     current_text_y),
                                fontFace=self.font,
                                fontScale=event_label_font_scale,
                                color=(220, 220, 220),
                                thickness=1,
                                lineType=cv2.LINE_AA
                            )
                        (text_width, text_height), text_baseline = cv2.getTextSize(event_label, self.font, event_label_font_scale, 2)
                        current_text_y += text_height + row_margin

            if self.panels['event_roll']['enable']:
                current_panel_text_y = self.panels['event_roll']['active_panel']['y']+3

                self.event_roll = self.move_image(image_data=self.event_roll, x=-self.event_roll_move, y=0)

                event_roll_slice_start = video_event_roll_offset + frame_index * self.event_roll_move
                event_roll_slice_end = video_event_roll_offset + (frame_index + 1) * self.event_roll_move

                if event_roll_slice_start >= video_event_roll.shape[1] or event_roll_slice_end >= video_event_roll.shape[1]:
                    event_roll_slice = numpy.zeros((video_event_roll.shape[0], self.event_roll_move, 3), numpy.uint8)
                else:
                    event_roll_slice = video_event_roll[:, event_roll_slice_start:event_roll_slice_end, :]

                if event_roll_slice.shape[1] > 0:
                    self.event_roll[:, -(self.event_roll_move + 1):-1, :] = event_roll_slice[:, :, :]

                if self.panels['event_roll']['header']:
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['event_roll']['header'], self.font, 0.8, 1)
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['event_roll']['header'],
                        org=(self.panels['event_roll']['top_x'], self.panels['event_roll']['top_y'] + text_height),
                        fontFace=self.font,
                        fontScale=0.8,
                        color=self.header_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['event_roll']['active_panel']['x'], self.panels['event_roll']['active_panel']['y']),
                    pt2=(self.panels['event_roll']['active_panel']['x'] + self.panels['event_roll']['active_panel']['size']['width'],
                         self.panels['event_roll']['active_panel']['y'] + self.panels['event_roll']['active_panel']['size']['height']),
                    color=self.panels['event_roll']['color'],
                    thickness=-1
                )

                if self.panels['event_roll']['frame_color']:
                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['event_roll']['active_panel']['x'], self.panels['event_roll']['active_panel']['y']),
                        pt2=(self.panels['event_roll']['active_panel']['x'] + self.panels['event_roll']['active_panel']['size']['width'],
                             self.panels['event_roll']['active_panel']['y'] + self.panels['event_roll']['active_panel']['size']['height']),
                        color=self.panels['event_roll']['frame_color'],
                        thickness=3
                    )

                # Place event roll
                offset_y = int((self.panels['event_roll']['active_panel']['size']['height'] - self.event_roll.shape[0]) / 2)
                offset_x = int((self.panels['event_roll']['active_panel']['size']['width'] - self.event_roll.shape[1]) / 2)

                self.place_image(
                    target_image=current_output_frame,
                    source_image=self.event_roll,
                    x=offset_x + self.panels['event_roll']['active_panel']['x'],
                    y=offset_y + self.panels['event_roll']['active_panel']['y']
                )

                col_id = 0
                shade = numpy.zeros_like(current_output_frame, numpy.uint8)

                cv2.rectangle(
                    img=shade,
                    pt1=(
                        self.panels['event_roll']['active_panel']['x'] + int(self.panels['event_roll']['active_panel']['size']['width'] / 2),
                        self.panels['event_roll']['active_panel']['y'] + 3
                    ),
                    pt2=(
                        self.panels['event_roll']['active_panel']['x'] + self.panels['event_roll']['active_panel']['size']['width'] - 3,
                        self.panels['event_roll']['active_panel']['y'] + self.panels['event_roll']['active_panel']['size']['height'] - 3
                    ),
                    color=(255, 255, 255),
                    thickness=cv2.FILLED
                )
                mask = shade.astype(bool)
                current_output_frame_shaded = current_output_frame.copy()
                alpha = 0.15
                current_output_frame_shaded[mask] = cv2.addWeighted(current_output_frame, alpha, shade, 1 - alpha, 0)[mask]
                current_output_frame = current_output_frame_shaded
                current_active_events = []
                for col_id, event_list_label in enumerate(self._event_lists):
                    current_active_events += self._event_lists[event_list_label].filter_time_segment(
                        start=frame_time,
                        stop=frame_time + self.frame_duration
                    ).unique_event_labels

                current_text_y = current_panel_text_y + int(label_lane_height / 2) + 6
                event_roll_height = self.panels['event_roll']['size']['height'] - 15 - text_baseline
                for scale in reversed(range(0, 100, 1)):
                    current_height = 0
                    current_width = 0
                    for col_id, event_label in enumerate(self.active_events):
                        column_max_width = 0
                        for event_label_id, event_label in enumerate(self.event_labels):
                            (text_width, text_height), text_baseline = cv2.getTextSize(event_label, fontFace=self.font, fontScale=scale / 10, thickness=1)
                            if col_id == 0:
                                current_height += text_height + int(text_height / 2)
                            if text_width > column_max_width:
                                column_max_width = text_width
                        current_width += column_max_width

                    if (current_height <= event_roll_height):
                        event_label_font_scale = scale / 10.0
                        break

                for event_label in self.active_events:
                    if event_label in current_active_events:
                        cv2.putText(
                            img=current_output_frame,
                            text=event_label,
                            org=(self.panels['event_roll']['active_panel']['x'] + int(self.panels['event_roll']['active_panel']['size']['width'] / 2),
                                 current_text_y),
                            fontFace=self.font,
                            fontScale=event_label_font_scale,
                            color=(50, 50, 50),
                            thickness=1,
                            lineType=cv2.LINE_AA
                        )
                    else:
                        cv2.putText(
                            img=current_output_frame,
                            text=event_label,
                            org=(self.panels['event_roll']['active_panel']['x'] + int(self.panels['event_roll']['active_panel']['size']['width'] / 2),
                                 current_text_y),
                            fontFace=self.font,
                            fontScale=event_label_font_scale,
                            color=(220, 220, 220),
                            thickness=1,
                            lineType=cv2.LINE_AA
                        )

                    col_id += 1
                    current_text_y += label_lane_height

                current_text_x = self.panels['event_roll']['top_x']
                current_text_y += 10
                for col_id, event_list_label in enumerate(self._event_lists):
                    # Event group title
                    (text_width, text_height), text_baseline = cv2.getTextSize(event_list_label, self.font, event_set_font_scale, 2)

                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(current_text_x , current_text_y-text_height),
                        pt2=(current_text_x +10, current_text_y),
                        color=event_list_colors[event_list_label],
                        thickness=-1
                    )

                    cv2.putText(
                        img=current_output_frame,
                        text=event_list_label,
                        org=(current_text_x + 20, current_text_y),
                        fontFace=self.font,
                        fontScale=event_set_font_scale,
                        color=(100, 100, 100),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    current_text_x += int(self.panels['event_roll']['size']['width'] / len(self._event_lists))

            if self.panels['event_text']['enable']:
                current_panel_text_y = self.panels['event_text']['top_y']

                # Labels
                column_margin = 20
                column_width = 200
                line_margin = 15
                event_label_font_scale = 0.5
                text_margin = 5

                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['event_text']['top_x'], self.panels['event_text']['top_y']),
                    pt2=(self.panels['event_text']['top_x'] + self.panels['event_text']['size']['width'], self.panels['event_text']['top_y'] + self.panels['event_text']['size']['height']),
                    color=self.panels['event_text']['color'],
                    thickness=-1
                )
                if self.panels['event_text']['frame_color']:
                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['event_text']['top_x'], self.panels['event_text']['top_y']),
                        pt2=(self.panels['event_text']['top_x'] + self.panels['event_text']['size']['width'],
                             self.panels['event_text']['top_y'] + self.panels['event_text']['size']['height']),
                        color=self.panels['event_text']['frame_color'],
                        thickness=3
                    )

                (text_width, text_height), text_baseline = cv2.getTextSize('text', self.font, event_set_font_scale, 2)
                event_list_height = self.panels['event_text']['size']['height'] - 15 - text_baseline

                event_label_font_scale = 1.0

                row_margin = int(text_height / 2)

                for col_id, event_list_label in enumerate(self._event_lists):
                    (text_width, text_height), text_baseline = cv2.getTextSize(text=event_list_label, fontFace=self.font, fontScale=event_set_font_scale, thickness=2)
                    if column_width < text_width:
                        column_width = text_width
                    for event_label_id, event_label in enumerate(self.event_labels):
                        (text_width, text_height), text_baseline = cv2.getTextSize(text=event_label, fontFace=self.font, fontScale=event_label_font_scale, thickness=1)
                        if column_width < text_width:
                            column_width = text_width

                for col_id, event_list_label in enumerate(self._event_lists):
                    current_text_y = current_panel_text_y + 15

                    current_active_events = self._event_lists[event_list_label].filter_time_segment(
                        start=frame_time,
                        stop=frame_time + self.frame_duration
                    ).unique_event_labels

                    # Event group title
                    (text_width, text_height), text_baseline = cv2.getTextSize(event_list_label, self.font, event_set_font_scale, 2)

                    current_output_frame = cv2.rectangle(
                        img=current_output_frame,
                        pt1=(self.panels['event_text']['top_x'] + col_id*(column_width + column_margin),
                             current_text_y-text_height-5),
                        pt2=(text_margin*2 + self.panels['event_text']['top_x'] + col_id*(column_width + column_margin)-2,
                             current_text_y+5),
                        color=event_list_colors[event_list_label],
                        thickness=-1
                    )
                    cv2.line(
                        img=current_output_frame,
                        pt1=(self.panels['event_text']['top_x'], current_text_y + int(text_baseline)+4),
                        pt2=(self.panels['event_text']['top_x'] + self.panels['event_text']['size']['width'],
                             current_text_y + int(text_baseline) + 4),
                        color=(200, 200, 200),
                        thickness=1
                    )

                    cv2.putText(
                        img=current_output_frame,
                        text=event_list_label,
                        org=(text_margin*2 + self.panels['event_text']['top_x'] + col_id*(column_width + column_margin),
                             current_text_y),
                        fontFace=self.font,
                        fontScale=event_set_font_scale,
                        color=(100, 100, 100),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                    current_text_y += text_baseline
                    (text_width, text_height), text_baseline = cv2.getTextSize(event_list_label, self.font, event_set_font_scale, 2)
                    current_text_y += text_baseline + row_margin * 4

                    active_labels = []
                    for event_label_id, event_label in enumerate(self.event_labels):
                        if event_label in current_active_events:
                            active_labels.append(event_label)

                    active_labels = textwrap.wrap(', '.join(active_labels), width=35)
                    for i, line in enumerate(active_labels):
                        textsize = cv2.getTextSize(line, self.font, event_label_font_scale, 1)[0]
                        gap = textsize[1] + line_margin

                        x = text_margin * 2 + self.panels['event_text']['top_x'] + col_id * (column_width + column_margin)
                        y = current_text_y + i * gap

                        cv2.putText(current_output_frame, line, (x, y), self.font,
                                    event_label_font_scale,
                                    (0, 0, 0),
                                    1,
                                    lineType=cv2.LINE_AA)

                    #cv2.putText(
                    #    img=current_output_frame,
                    #    text=active_labels,
                    #    org=(text_margin*2 + self.panels['event_list']['top_x'] + col_id * (column_width + column_margin),
                    #         current_text_y),
                    #    fontFace=self.font,
                    #    fontScale=event_label_font_scale,
                    #    color=(0, 0, 0),
                    #    thickness=1,
                    #    lineType=cv2.LINE_AA
                    #)

                    (text_width, text_height), text_baseline = cv2.getTextSize(event_label, self.font, event_label_font_scale, 2)
                    current_text_y += text_height + row_margin


            if self.panels['footer']['enable']:
                # Footer
                current_output_frame = cv2.rectangle(
                    img=current_output_frame,
                    pt1=(self.panels['footer']['top_x'], self.panels['footer']['top_y'] - 1),
                    pt2=(self.panels['footer']['top_x'] + self.panels['footer']['size']['width'], self.panels['footer']['top_y'] + self.panels['footer']['size']['height']),
                    color=self.panels['footer']['color'],
                    thickness=-1
                )
                if self.panels['footer']['header']:
                    (text_width, text_height), text_baseline = cv2.getTextSize(self.panels['footer']['header'], self.font, 0.8, 2)
                    cv2.putText(
                        img=current_output_frame,
                        text=self.panels['footer']['header'],
                        org=(self.panels['footer']['top_x'] + int(self.panels['footer']['size']['width'] / 2) - int(text_width/2), self.panels['footer']['top_y'] + int(self.panels['footer']['size']['height'] / 2) + text_height),
                        fontFace=self.font,
                        fontScale=0.8,
                        color=self.header_color,
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                if 'footer_left' in self.logos:
                    self.place_image_transparent(
                        target_image=current_output_frame,
                        source_image=self.logos['footer_left'],
                        x=self.panels['footer']['top_x']+4,
                        y=self.panels['footer']['top_y']+3
                    )

                if 'footer_right' in self.logos:
                    self.place_image_transparent(
                        target_image=current_output_frame,
                        source_image=self.logos['footer_right'],
                        x=self.panels['footer']['top_x']+self.panels['footer']['size']['width']-self.logos['footer_right'].shape[1]-4,
                        y=self.panels['footer']['top_y']+3
                    )

            # Wait for keypress
            k = cv2.waitKey(10)
            if k == ord('q'):
                break
            if not video_ret:
                break

            # Display the image
            cv2.imshow('output', current_output_frame)

            # Write the frame
            self.output.write(current_output_frame)

            frame_index += 1
            frame_time += self.frame_duration

        self.video_data.release()
        self.output.release()
        cv2.destroyAllWindows()

        import ffmpeg
        import shutil

        from pathlib import Path
        p = Path(self.target_video)
        target_video_tmp =  str(p.with_suffix('.tmp.mp4'))

        if self.audio_only:
            input_video = ffmpeg.input(self.target_video)
            input_audio = ffmpeg.input(self.source)
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(target_video_tmp, **{'qscale:v': 3}).run(overwrite_output=True)
            shutil.copyfile(target_video_tmp, self.target_video)
            os.remove(target_video_tmp)
        else:
            input_video = ffmpeg.input(self.target_video)
            input_audio = ffmpeg.input(self.source)
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(self.target_video).run(overwrite_output=True)
