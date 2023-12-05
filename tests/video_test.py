#!/usr/bin/env python
import sed_vis
import dcase_util
import os

current_path = os.path.dirname(os.path.realpath(__file__))

video_file = os.path.join('data', 'street_traffic-london-271-8243.mp4')

event_lists = {
    'Reference': dcase_util.containers.MetaDataContainer().load(
        os.path.join(current_path, 'data', 'street_traffic-london-271-8243.ann')
    ),
    'Baseline': dcase_util.containers.MetaDataContainer().load(
        os.path.join(current_path, 'data', 'street_traffic-london-271-8243_sys1.ann')
    ),
    'Proposed': dcase_util.containers.MetaDataContainer().load(
        os.path.join(current_path, 'data', 'street_traffic-london-271-8243_sys2.ann')
    )
}

generator = sed_vis.video.VideoGenerator(
    source_video=video_file,
    target_video=os.path.join('data', 'street_traffic-london-271-8243.output.mp4'),
    event_lists=event_lists,
    event_list_order=['Reference', 'Baseline', 'Proposed'],

    title='Test video',
    intro_text='This is a test video generation.',
    logos={
        'left': os.path.join(current_path,'logo.png'),
        'right': os.path.join(current_path,'logo.png')
    },
    layout=[
        ['video', 'spectrogram'],
        ['event_list', 'event_roll'],
        ['footer']
    ]
).generate()
