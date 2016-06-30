#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
I/O
==================
Functions for loading in annotations from files in various formats.

.. autosummary::
    :toctree: generated/

    load_audio
"""

import os
import numpy
import wave
import csv
import subprocess
import scipy.signal
import util.event_list


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array

    Supports 24-bit wav-format, other audio formats are supported through FFMPEG.

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        audio_file = wave.open(filename)

        # Audio info
        sample_rate = audio_file.getframerate()
        sample_width = audio_file.getsampwidth()
        number_of_channels = audio_file.getnchannels()
        number_of_frames = audio_file.getnframes()

        # Read raw bytes
        data = audio_file.readframes(number_of_frames)
        audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = numpy.empty((num_samples, number_of_channels, 4), dtype=numpy.uint8)
            raw_bytes = numpy.fromstring(data, dtype=numpy.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = numpy.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        # Down-mix audio
        if mono:
            audio_data = numpy.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            n_samples = int(numpy.ceil(audio_data.shape[-1] * float(fs) / sample_rate))
            audio_data = scipy.signal.resample(audio_data, n_samples, axis=-1)
            audio_data = numpy.ascontiguousarray(audio_data, dtype=audio_data.dtype)

    else:
        command = ['ffmpeg',
                   '-v', 'quiet',
                   '-i', '-',  # audio_filename,
                   '-f', 'f32le',
                   '-ar', str(fs),  # output will have fs
                   '-ac', '1',  # stereo (set to '1' for mono)
                   '-']
        audio_data = numpy.empty(0)
        with open(filename, "rb") as afilename:
            pipe = subprocess.Popen(command, stdin=afilename,
                                    stdout=subprocess.PIPE, bufsize=10 ** 8)
            while True:
                raw_audio = pipe.stdout.read(88200 * 4)
                if len(raw_audio) == 0:
                    break

                audio_array = numpy.fromstring(raw_audio, dtype="float32")
                audio_data = numpy.hstack((audio_data, audio_array))

        pipe = subprocess.Popen(['ffmpeg', '-i', filename, '-'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pipe.stdout.readline()
        pipe.terminate()
        infos = pipe.stderr.read()

    return audio_data, fs


def load_event_list(filename):
    """Load event list from csv formatted text-file

    Supported formats:

    - [event onset (float >= 0)][delimiter][event offset (float >= 0)]
    - [event onset (float >= 0)][delimiter][event offset (float >= 0)][delimiter][label]
    - [filename][delimiter][event onset (float >= 0)][delimiter][event offset (float >= 0)][delimiter][event label]
    - [filename][delimiter][scene_label][delimiter][event onset (float >= 0)][delimiter][event offset (float >= 0)][delimiter][event label]

    Supported delimiters: ``,``, ``;``, ``tab``

    Example of event list file::

        21.64715	23.00552	alert
        36.91184	38.27021	alert
        69.72575	71.09029	alert
        63.53990	64.89827	alert
        84.25553	84.83920	alert
        20.92974	21.82661	clearthroat
        28.39992	29.29679	clearthroat
        80.47837	81.95937	clearthroat
        44.48363	45.96463	clearthroat
        78.13073	79.05953	clearthroat
        15.17031	16.27235	cough
        20.54931	21.65135	cough
        27.79964	28.90168	cough
        75.45959	76.32490	cough
        70.81708	71.91912	cough
        21.23203	22.55902	doorslam
        7.546220	9.014880	doorslam
        34.11303	35.04183	doorslam
        45.86001	47.32867	doorslam


    Parameters
    ----------
    filename : str
        Path to the csv-file

    Returns
    -------
    event_list: list
        Event list

    """

    data = []
    file_open = False
    if hasattr(filename, 'read'):
        input_file = filename
    else:
        input_file = open(filename, 'rt')
        file_open = True

    try:
        dialect = csv.Sniffer().sniff(input_file.readline(), [',', ';', '\t'])
    except csv.Error:
        raise ValueError('Unknown delimiter.')

    input_file.seek(0)

    for row in csv.reader(input_file, dialect):
        if len(row):
            if len(row) == 2:
                if not isfloat(row[0]) or not isfloat(row[1]):
                    raise ValueError('Event onset and event offset needs to be float.')

                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1])
                    }
                )
            elif len(row) == 3:
                if not isfloat(row[0]) or not isfloat(row[1]):
                    raise ValueError('Event onset and event offset needs to be float.')

                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1]),
                        'event_label': row[2]
                    }
                )
            elif len(row) == 4:
                if not isfloat(row[1]) or not isfloat(row[2]):
                    raise ValueError('Event onset and event offset needs to be float.')

                data.append(
                    {
                        'file': row[0],
                        'event_onset': float(row[1]),
                        'event_offset': float(row[2]),
                        'event_label': row[3]
                    }
                )
            elif len(row) == 5:
                if not isfloat(row[2]) or not isfloat(row[3]):
                    raise ValueError('Event onset and event offset needs to be float.')

                data.append(
                    {
                        'file': row[0],
                        'scene_label': row[1],
                        'event_onset': float(row[2]),
                        'event_offset': float(row[3]),
                        'event_label': row[4]
                    }
                )
            else:
                raise ValueError('Unknown event list format.')

    if file_open:
        input_file.close()

    return util.event_list.EventList(data)


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False