#!/usr/bin/env python
import sed_vis
import dcase_util
import os

current_path = os.path.dirname(os.path.realpath(__file__))

generator = sed_vis.video.VideoGenerator(
    source_video=os.path.join('data', 'street_traffic-london-271-8243.mp4'),
    source_audio=os.path.join('data', 'street_traffic-london-271-8243.mp4'),
    target=os.path.join('data', 'street_traffic-london-271-8243.output.mp4'),
    event_lists={
        'Reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'street_traffic-london-271-8243.ann')
        ),
        'Baseline': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'street_traffic-london-271-8243.ann')
        ),
        'Proposed': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'street_traffic-london-271-8243_sys2.ann')
        )
    },
    event_list_order=['Reference', 'Baseline', 'Proposed'],
    logos={
        'footer_right': os.path.join(current_path,'logo.png')
    },
    spectrogram_height=250,
    text={
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
    },
    layout=[
        # ['header'],
        # ['video'],
        #['video', 'spectrogram'],
        ['spectrogram', 'video'],
        #['spectrogram'],
        ['mid_header'],
        #['video_dummy', 'event_roll'],
        ['event_roll', 'video_dummy'],
        #['event_list'], #, 'event_roll'],
        #['event_roll'],
        #['event_text'],
        #['footer']
    ]
).generate()
