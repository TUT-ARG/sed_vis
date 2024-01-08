#!/usr/bin/env python
import sed_vis
import dcase_util
import os

current_path = os.path.dirname(os.path.realpath(__file__))

generator = sed_vis.video.VideoGenerator(
    source=os.path.join('data', 'a001.wav'), # os.path.join('data', 'street_traffic-london-271-8243.mp4') #
    target=os.path.join('data', 'a001.output.mp4'), # os.path.join('data', 'street_traffic-london-271-8243.output.mp4')
    event_lists={
        'Reference': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann') # os.path.join(current_path, 'data', 'street_traffic-london-271-8243.ann')
        ),
        'Baseline': dcase_util.containers.MetaDataContainer().load(
            os.path.join(current_path, 'data', 'a001.ann') # os.path.join(current_path, 'data', 'street_traffic-london-271-8243_sys1.ann')
        ),
        #'Proposed': dcase_util.containers.MetaDataContainer().load(
        #    os.path.join(current_path, 'data', 'street_traffic-london-271-8243_sys2.ann')
        #)
    },
    event_list_order=['Reference', 'Baseline'], #, 'Proposed'],

    logos={
        'header_left': os.path.join(current_path,'logo.png'),
        'footer_right': os.path.join(current_path,'logo.png')
    },
    #headers={
    #    'header': 'Header',
    #    'video': 'Video stream',
    #    'spectrogram': 'Spectrogram',
    #    'event_list': 'Events',
    #    'event_roll': 'Events',
    #    'footer': 'Footer'
    #},
    layout=[
        # ['header'],
        # ['video'],
        #['video', 'spectrogram'],
        ['spectrogram'],
        ['mid_header'],
        # ['video_dummy', 'event_roll'],
        #['event_list', 'event_roll'],
        ['event_roll'],
        ['footer']
    ]
).generate()
