#!/usr/bin/env python
import sed_vis
import os

mode = 'multiple' #''dcase2016'
current_path = os.path.dirname(os.path.realpath(__file__))

if mode == 'dcase2016':
    pass
    audio, fs = sed_vis.io.load_audio(os.path.join(current_path, 'data/sound_event/dcase2016/a030_mono_16.wav'))
    vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list(os.path.join(current_path,'data/sound_event/dcase2016/a030.ann'))},
                                                    event_list_order=['reference'],
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    spec_cmap='jet',
                                                    spec_interpolation='bicubic',
                                                    spec_win_size=1024,
                                                    spec_hop_size=1024/2,
                                                    spec_fft_size=1024,
                                                    publication_mode=True)
    vis.show()

elif mode == 'publication':
    # Example how to create plots for publications, use "save the figure" button and
    # select svg format. Open figure in e.g. inkscape and edit to your liking.
    audio, fs = sed_vis.io.load_audio(os.path.join(current_path, 'data/a001.wav'))
    vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001.ann')),
                                                                 'full': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001_full.ann')),
                                                                 'estimated': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001_system_output.ann'))},
                                                    event_list_order=['reference', 'full', 'estimated'],
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    spec_cmap='jet',
                                                    spec_interpolation='bicubic',
                                                    spec_win_size=1024,
                                                    spec_hop_size=1024/8,
                                                    spec_fft_size=1024,
                                                    publication_mode=True)
    vis.show()

elif mode == 'sync':
    # Test for audio and visual synchronization during the playback.
    audio, fs = sed_vis.io.load_audio(os.path.join(current_path, 'data/sync/sin_silence.wav'))
    vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list(os.path.join(current_path,'data/sync/sin_silence.txt'))},
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    mode='time_domain'
                                                    )

    vis.show()

elif mode == 'multiple':
    # Test visualization of multiple system outputs
    audio, fs = sed_vis.io.load_audio(os.path.join(current_path, 'data/a001.wav'))
    vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001.ann')),
                                                                 'estimated1': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001_system_output.ann')),
                                                                 'estimated2': sed_vis.io.load_event_list(os.path.join(current_path,'data/a001_system_output_2.ann'))},
                                                    event_list_order=['reference', 'estimated1', 'estimated2'],
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    spec_cmap='jet',
                                                    spec_interpolation='bicubic',
                                                    spec_win_size=1024,
                                                    spec_hop_size=1024/8,
                                                    spec_fft_size=1024,
                                                    publication_mode=True)
    vis.show()