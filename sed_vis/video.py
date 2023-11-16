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
import matplotlib

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

        self.source_video = kwargs.get('source_video', None)
        self.target_video = kwargs.get('target_video', None)

        self.title = kwargs.get('title', None)
        self.intro_text = kwargs.get('intro_text', None)

        self.video_data = cv2.VideoCapture(filename=self.source_video)

        self.audio_data = dcase_util.containers.AudioContainer().load(filename=self.source_video, mono=True, fs=16000)
        self.audio_info = dcase_util.utils.get_audio_info(filename=self.source_video)
        self.fs = self.audio_data.fs

        self.frame_width = kwargs.get('frame_width', 1280)
        self.frame_height = kwargs.get('frame_height', 720)

        self.fps = self.video_data.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1.0 / self.fps
        self.frame_count = int(self.video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        self.output = cv2.VideoWriter(self.target_video, fourcc,  self.fps, (self.frame_width, self.frame_height))

        self.panels = {
            'footer': {
                'size': {
                    'height': 40
                },
                'top_y': 680,
                'top_x': 0
            },
            'header': {
                'size': {
                    'height': 474
                },
                'top_y': 0,
                'top_x': 0
            },
            'video': {
                'size': {
                  'width': 400,
                  'height': 300
                },
                'top_y': 30,
                'top_x': 10
            },
            'spectrogram': {
                'size': {
                    'width': 660,
                    'height': 300,
                },
                'top_y': 30,
                'top_x': None
            }
        }
        self.panels['footer']['size']['width'] = self.frame_width
        self.panels['header']['size']['width'] = self.frame_width

        self.panels['spectrogram']['top_x'] = self.panels['video']['size']['width'] + 50
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.spectrogram_move = 10
        self.spectrogram_segment_duration = self.panels['spectrogram']['size']['width'] / self.spectrogram_move * self.frame_duration
        self.cmap = matplotlib.cm.get_cmap('magma')
        self.norm = matplotlib.colors.Normalize(vmin=-20, vmax=20)

        self.spectrogram = numpy.zeros((self.panels['spectrogram']['size']['height']-6, self.panels['spectrogram']['size']['width']-6, 3), numpy.uint8)
        self.spectrogram.fill(0)
        self.fft_size = 1024

        self.ui.section_header('Video generator')
        self.ui.data('Source', self.source_video)
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
    def place_image(target_image, source_image, x, y):
        target_image[y:y + source_image.shape[0],x:x + source_image.shape[1]] = source_image

    def generate(self):
        import cv2
        import librosa

        from IPython import embed

        frame_index = 0
        frame_time = 0

        self.audio_data.normalize()
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

        video_spectrogram = cv2.resize(
            src=video_spectrogram,
            dsize=(self.frame_count*self.spectrogram_move, self.panels['spectrogram']['size']['height'] - 6),
            interpolation=cv2.INTER_AREA
        )

        video_spectrogram = numpy.flip(video_spectrogram)

        video_spectrogram_offset = int(self.spectrogram.shape[1]/2)
        self.spectrogram[:, video_spectrogram_offset:, :] = video_spectrogram[:, 0:video_spectrogram_offset, :]

        while True:
            self.spectrogram = self.move_image(image_data=self.spectrogram, x=-self.spectrogram_move, y=0)

            spectrogram_slice_start = video_spectrogram_offset + frame_index * self.spectrogram_move
            spectrogram_slice_end = video_spectrogram_offset + (frame_index+1) * self.spectrogram_move

            if spectrogram_slice_start >= video_spectrogram.shape[1] or spectrogram_slice_end >= video_spectrogram.shape[1]:
                spectrogram_slice = numpy.zeros((video_spectrogram.shape[0], self.spectrogram_move, 3), numpy.uint8)
            else:
                spectrogram_slice = video_spectrogram[:, spectrogram_slice_start:spectrogram_slice_end, :]

            if spectrogram_slice.shape[1] > 0:
                self.spectrogram[:, -(self.spectrogram_move + 1):-1, :] = spectrogram_slice[:, :, :]

            video_ret, video_frame = self.video_data.read()
            if video_ret:
                video_frame = self.image_resize(
                    image_data=video_frame,
                    height=self.panels['video']['size']['height']-(3*2),
                    width=self.panels['video']['size']['width']-(3*2),
                )

            current_output_frame = numpy.zeros((self.frame_height, self.frame_width, 3), numpy.uint8)
            current_output_frame.fill(255)

            # Layout
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['header']['top_x'], self.panels['header']['top_y'] - 1),
                pt2=(self.panels['header']['top_x'] + self.panels['header']['size']['width'], self.panels['header']['top_y'] + self.panels['header']['size']['height']),
                color=(240, 240, 240),
                thickness=-1
            )
            #current_output_frame = cv2.rectangle(
            #    img=current_output_frame,
            #    pt1=(0, 0),
            #    pt2=(self.frame_width, 474),
            #    color=(240, 240, 240),
            #    thickness=-1
            #)

            # Footer
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['footer']['top_x'], self.panels['footer']['top_y'] - 1),
                pt2=(self.panels['footer']['top_x'] + self.panels['footer']['size']['width'], self.panels['footer']['top_y'] + self.panels['footer']['size']['height']),
                color=(240, 240, 240),
                thickness=-1
            )

            # Video box
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['video']['top_x'], self.panels['video']['top_y']),
                pt2=(self.panels['video']['top_x'] + self.panels['video']['size']['width'], self.panels['video']['top_y'] + self.panels['video']['size']['height']),
                color=(0, 0, 0),
                thickness=-1
            )
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['video']['top_x'], self.panels['video']['top_y']),
                pt2=(self.panels['video']['top_x'] + self.panels['video']['size']['width'], self.panels['video']['top_y'] + self.panels['video']['size']['height']),
                color=(100, 100, 100),
                thickness=3
            )

            # Spectrogram box
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['spectrogram']['top_x'], self.panels['spectrogram']['top_y']),
                pt2=(self.panels['spectrogram']['top_x'] + self.panels['spectrogram']['size']['width'], self.panels['spectrogram']['top_y'] + self.panels['spectrogram']['size']['height']),
                color=(0, 0, 0),
                thickness=-1
            )
            current_output_frame = cv2.rectangle(
                img=current_output_frame,
                pt1=(self.panels['spectrogram']['top_x'], self.panels['spectrogram']['top_y']),
                pt2=(self.panels['spectrogram']['top_x'] + self.panels['spectrogram']['size']['width'], self.panels['spectrogram']['top_y'] + self.panels['spectrogram']['size']['height']),
                color=(100, 100, 100),
                thickness=3
            )


            if video_ret:
                # Place video frame
                offset_y = int((self.panels['video']['size']['height'] - video_frame.shape[0])/2)
                offset_x = int((self.panels['video']['size']['width'] - video_frame.shape[1])/2)
                self.place_image(
                    target_image=current_output_frame,
                    source_image=video_frame,
                    x=offset_x + self.panels['video']['top_x'],
                    y=offset_y + self.panels['video']['top_y']
                )

            # Place spectrogram
            offset_y = int((self.panels['spectrogram']['size']['height'] - self.spectrogram.shape[0]) / 2)
            offset_x = int((self.panels['spectrogram']['size']['width'] - self.spectrogram.shape[1]) / 2)
            self.place_image(
                target_image=current_output_frame,
                source_image=self.spectrogram,
                x=offset_x + self.panels['spectrogram']['top_x'],
                y=offset_y + self.panels['spectrogram']['top_y']
            )

            freq_index = 0
            tick_hz = 16000 / 4.0
            for y in range(self.panels['spectrogram']['top_y'], self.panels['spectrogram']['top_y'] + self.spectrogram.shape[0] + 1, int(self.spectrogram.shape[0] / 4)):
                cv2.putText(
                    img=current_output_frame,
                    text='{index:.0f}kHz'.format(index=(16000 - freq_index * tick_hz) / 1000),
                    org=(self.panels['spectrogram']['top_x'] + self.spectrogram.shape[1] + 10, y + 4),
                    fontFace=self.font,
                    fontScale=0.5,
                    color=(100, 100, 100),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                freq_index += 1

            cv2.putText(
                img=current_output_frame,
                text='Frequency',
                org=(self.panels['spectrogram']['top_x'] + self.spectrogram.shape[1] + 55, self.panels['spectrogram']['top_y'] + 4 + int(self.spectrogram.shape[0] / 2)),
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
                    self.panels['spectrogram']['top_x'] + int(self.panels['spectrogram']['size']['width']/2),
                    self.panels['spectrogram']['top_y'] + 3
                ),
                pt2=(
                    self.panels['spectrogram']['top_x'] + self.panels['spectrogram']['size']['width']-3,
                    self.panels['spectrogram']['top_y'] + self.panels['spectrogram']['size']['height']-3
                ),
                color=(255, 255, 255),
                thickness=cv2.FILLED
            )
            mask = shade.astype(bool)
            current_output_frame_shaded = current_output_frame.copy()
            alpha = 0.5
            current_output_frame_shaded[mask] = cv2.addWeighted(current_output_frame, alpha, shade, 1 - alpha, 0)[mask]

            # Wait for keypress
            k = cv2.waitKey(10)
            if k == ord('q'):
                break
            if not video_ret:
                break

            # Display the image
            cv2.imshow('output', current_output_frame_shaded)

            # Write the frame
            self.output.write(current_output_frame_shaded)

            frame_index += 1
            frame_time += self.frame_duration

        self.video_data.release()
        self.output.release()
        cv2.destroyAllWindows()

        #embed()