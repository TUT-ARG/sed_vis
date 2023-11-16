#!/usr/bin/env python
import sed_vis
import dcase_util
import os

current_path = os.path.dirname(os.path.realpath(__file__))

video_file = os.path.join('data', 'street_traffic-london-271-8243.mp4')

event_lists = {
    'reference': dcase_util.containers.MetaDataContainer().load(
        os.path.join(current_path, 'data', 'street_traffic-london-271-8243.ann')
    )
}

generator = sed_vis.video.VideoGenerator(
    source_video=video_file,
    target_video=os.path.join('data', 'street_traffic-london-271-8243.output.mp4'),
    event_lists=event_lists,
    event_list_order=['reference'],

    title='Test video',
    intro_text='This is a test video generation.',
).generate()
