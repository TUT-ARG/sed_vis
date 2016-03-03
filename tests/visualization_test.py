import sed_vis

audio, fs = sed_vis.io.load_audio('data/a001.wav')
vis = sed_vis.visualization.EventListVisualizer(event_lists={'reference': sed_vis.io.load_event_list('data/a001.ann'),
                                                             'full': sed_vis.io.load_event_list('data/a001_full.ann'),
                                                             'estimated': sed_vis.io.load_event_list('data/a001_system_output.ann')},
                                                audio_signal=audio,
                                                sampling_rate=fs)


vis.show()
