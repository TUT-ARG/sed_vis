#!/usr/bin/env python
import sed_vis
import dcase_util
import os

mode = 'probability'
current_path = os.path.dirname(os.path.realpath(__file__))

if mode == 'dcase2016':
    audio_container = dcase_util.containers.AudioContainer().load(
        os.path.join(current_path, 'data', 'a001.wav')
    )

    event_lists = {
        'reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann')
        )
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=['reference'],
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        spec_cmap='jet',
        spec_interpolation='bicubic',
        spec_win_size=1024,
        spec_hop_size=1024/2,
        spec_fft_size=1024,
        publication_mode=True
    )

    vis.show()

elif mode == 'publication':
    # Example how to create plots for publications, use "save the figure" button and
    # select svg format. Open figure in e.g. inkscape and edit to your liking.
    audio_container = dcase_util.containers.AudioContainer().load(
        os.path.join(current_path, 'data', 'a001.wav')
    )
    event_lists = {
        'reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann')
        ),
        'full': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001_full.ann')
        ),
        'estimated': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001_system_output.ann')
        )
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=['reference', 'full', 'estimated'],
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        spec_cmap='jet',
        spec_interpolation='bicubic',
        spec_win_size=1024,
        spec_hop_size=1024/8,
        spec_fft_size=1024,
        publication_mode=True
    )

    vis.show()

elif mode == 'sync':
    # Test for audio and visual synchronization during the playback.
    audio_container = dcase_util.containers.AudioContainer().load(
        os.path.join(current_path, 'data', 'sync', 'sin_silence.wav')
    )

    event_lists = {
        'reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'sync', 'sin_silence.txt')
        )
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        mode='time_domain'
    )

    vis.show()

elif mode == 'multiple':
    # Test visualization of multiple system outputs
    audio_container = dcase_util.containers.AudioContainer().load(
        os.path.join(current_path, 'data', 'a001.wav')
    )

    event_lists = {
        'reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann')
        ),
        'estimated1': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001_system_output.ann')
        ),
        'estimated2': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001_system_output_2.ann')
        )
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=['reference', 'estimated1', 'estimated2'],
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        spec_cmap='jet',
        spec_interpolation='bicubic',
        spec_win_size=1024,
        spec_hop_size=1024/8,
        spec_fft_size=1024,
        publication_mode=True
    )

    vis.show()

elif mode == 'probability':
    audio_container = dcase_util.containers.AudioContainer().load(
        os.path.join(current_path, 'data', 'a001.wav')
    )
    event_lists = {
        'reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann')
        ),
        'estimated': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001_system_output_prob.csv')
        )
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=['reference','estimated'], # 'full', 'estimated'],
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        spec_cmap='jet',
        spec_interpolation='bicubic',
        spec_win_size=1024,
        spec_hop_size=1024/8,
        spec_fft_size=1024,
        publication_mode=True
    )

    vis.show()