#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio player
==================

.. autosummary::
    :toctree: generated/

    AudioPlayer
    AudioPlayer.play
    AudioPlayer.stop
    AudioPlayer.pause
    AudioPlayer.get_time

"""

import numpy
import threading
from numpy.lib.stride_tricks import as_strided

class AudioPlayer(object):
    def __init__(self, signal, sampling_rate):
        self.signal = signal
        self.sampling_rate = sampling_rate

        if len(self.signal.shape) == 1:
            self.channels = 1
        else:
            self.channels = self.signal.shape[1]

        self.player_thread = None
        self.play_start_time = 0

        import pyaudio
        self._pa = pa = pyaudio.PyAudio()
        self._threads = []

        # Lockers
        self.halting = threading.Lock()  # Only for "close" method
        self.lock = threading.Lock()  # "_threads" access locking
        self.finished = False
        self.playing = False

    def __del__(self):
        self.close()

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __enter__(self):
        return self

    @property
    def fs(self):
        return self.sampling_rate

    @property
    def duration_seconds(self):
        return self.signal.shape[0] / float(self.sampling_rate)

    @property
    def duration_samples(self):
        return self.signal.shape[0]

    def close(self):
        """
        Destructor for this audio interface. Waits the threads to finish their
        streams, if desired.
        """
        with self.halting:  # Avoid simultaneous "close" threads

            if not self.finished:  # Ignore all "close" calls, but the first,
                self.finished = True  # and any call to play would raise ThreadError

                # Closes all playing AudioThread instances
                while True:
                    with self.lock:  # Ensure there's no other thread messing around
                        try:
                            thread = self._threads[0]  # Needless to say: pop = deadlock
                        except IndexError:  # Empty list
                            break  # No more threads

                    thread.stop()
                    thread.join()

                # Finishes
                assert not self._pa._streams  # No stream should survive
                self._pa.terminate()

    def terminate(self):
        self.close()  # Avoids direct calls to inherited "terminate"

    def play(self, offset=0.0, duration=None):
        """
        Start another thread playing the given audio sample iterable (e.g. a
        list, a generator, a NumPy np.ndarray with samples), and play it.
        The arguments are used to customize behaviour of the new thread, as
        parameters directly sent to PyAudio's new stream opening method, see
        AudioThread.__init__ for more.
        """
        if self.playing:
            # If playback is on, stop play
            self.stop()

        with self.lock:
            if self.finished:
                raise threading.ThreadError("Trying to play an audio stream while "
                                            "halting the AudioIO manager object")
            self.player_thread = AudioThread(device_manager=self,
                                             audio=self.get_segment(offset, duration),
                                             chunk_size=2048,
                                             sampling_rate=self.sampling_rate,
                                             nchannels=self.channels)

            self.player_thread.start()
            self.playing = True

    def get_segment(self, offset=0.0, duration=None, minimum_duration_samples=4096):
        start_id = int(offset * self.sampling_rate)
        if duration:
            stop_id = int((offset + duration) * self.sampling_rate)
            if stop_id > self.signal.shape[0]:
                stop_id = self.signal.shape[0]
        else:
            stop_id = self.signal.shape[0]

        if stop_id - start_id < minimum_duration_samples:
            n = numpy.zeros((minimum_duration_samples))
            a = self.signal[start_id:stop_id]
            n[0:a.shape[0]] = a
            return n
        else:
            return self.signal[start_id:stop_id]

    def stop(self):
        if self.playing:
            self.player_thread.stop()
            self.playing = False
            self.player_thread = None

    def pause(self):
        if self.playing:
            self.player_thread.pause()
            self.playing = False

    def get_time(self):
        if self.playing:
            return self.player_thread.time
        else:
            return 0


class AudioThread(threading.Thread):
    """
    Audio playback thread

    After audiolazy.audioIO
    """

    def __init__(self, device_manager, audio,
                 chunk_size,
                 channels=1,
                 sampling_rate=44100,
                 **kwargs
                 ):
        """
        Sets a new thread to play the given audio.

        Parameters
        ----------
        chunk_size : int
            Number of samples per chunk (block sent to device).

        channels : int [>=1]
            Channels in audio stream (serialized).
            (Default value=1)

        sampling_rate : int
            Sampling rate
            (Default value=44100)

        """
        super(AudioThread, self).__init__()

        # Stores data needed by the run method
        self.audio = audio

        self.device_manager = device_manager
        self.nchannels = kwargs.pop("nchannels", channels)
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate

        # Split audio into chunks already before playback
        if self.chunk_size < 1:
            raise ParameterError('Invalid chunk size: {:d}'.format(self.chunk_size))

        # Compute the number of frames, end may be truncated.
        n_frames = 1 + int((len(self.audio) - self.chunk_size) / self.chunk_size)

        if n_frames < 1:
            raise ParameterError('Buffer is too short')

        # Vertical stride is one sample
        # Horizontal stride is `hop_length` samples
        self.chunks = as_strided(self.audio, shape=(self.chunk_size, n_frames),
                                 strides=(self.audio.itemsize, self.chunk_size * self.audio.itemsize))

        # Lockers
        self.lock = threading.Lock()  # Avoid control methods simultaneous call
        self.go = threading.Event()   # Communication between the 2 threads
        self.go.set()
        self.halting = False          # The stop message

        # Get the streaming function
        import _portaudio  # Just to be slightly faster (per chunk!)
        self.write_stream = _portaudio.write_stream

        # Open a new audio output stream
        self.stream = device_manager._pa.open(format=1,
                                              channels=self.nchannels,
                                              rate=sampling_rate,
                                              frames_per_buffer=self.chunk_size,
                                              output=True,
                                              **kwargs)
        self.time = 0

    def run(self):
        """Audio playback
        """
        # From now on, it's multi-thread. Let the force be with them.
        st = self.stream._stream
        i = 0
        for chunk_id in range(0, self.chunks.shape[1]):
            current_chunk = self.chunks[:, chunk_id].astype(numpy.float32).tostring()
            self.write_stream(st, current_chunk, self.chunk_size, False)

            if not self.go.is_set():
                self.stream.stop_stream()
                if self.halting:
                    break
                self.go.wait()
                self.stream.start_stream()

            i += self.chunk_size * self.nchannels
            self.time = (i / self.nchannels) / float(self.sampling_rate)

        self.stream.close()
        self.device_manager.stop()

    def stop(self):
        """ Stops the playing thread and close """
        with self.lock:
            self.halting = True
            self.go.clear()

    def pause(self):
        """ Pauses the audio. """
        with self.lock:
            self.go.clear()

    def play(self):
        """ Resume playing the audio. """
        with self.lock:
            self.go.set()
