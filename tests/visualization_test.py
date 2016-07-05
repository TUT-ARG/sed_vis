import sed_vis
import os

mode = 'publication'

if mode == 'publication':
    # Example how to create plots for publications, use "save the figure" button and
    # select svg format. Open figure in e.g. inkscape and edit to your liking.
    current_path = os.path.dirname(os.path.realpath(__file__))
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
                                                    spec_fft_size=1024)
    vis.show()

elif mode == 'sync':
    # Test for audio and visual synchronization during the playback.
    current_path = os.path.dirname(os.path.realpath(__file__))
    audio, fs = sed_vis.io.load_audio(os.path.join(current_path, 'data/sync/sin_silence.wav'))
    vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list(os.path.join(current_path,'data/sync/sin_silence.txt'))},
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    mode='time_domain'
                                                    )

    vis.show()
