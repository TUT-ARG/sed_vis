#!/usr/bin/env python
"""
Visualizer for sound event detection system
"""

import sys
import os
import argparse
import textwrap
import sed_vis
import dcase_util

__version_info__ = ('0', '1', '0')
__version__ = '.'.join(__version_info__)


def process_arguments(argv):

    # Argparse function to get the program parameters
    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Sound Event Visualizer
        '''))

    # Setup argument handling
    parser.add_argument('-a',
                        dest='audio_file',
                        default=None,
                        type=str,
                        action='store',
                        help='<Required> Audio file',
                        required=True)

    parser.add_argument('-l',
                        '--list',
                        nargs='+',
                        help='<Required> List of event list files',
                        required=True)

    parser.add_argument('-n',
                        '--names',
                        nargs='+',
                        help='List of names for event lists files (same order than event list files)',
                        required=False)

    parser.add_argument('-e',
                        '--events',
                        nargs='+',
                        help='List of active event classes',
                        required=False)

    parser.add_argument('--time_domain',
                        help="Time domain visualization",
                        action="store_true")

    parser.add_argument('--spectrogram',
                        help="Spectrogram visualization <default>",
                        action="store_true")

    parser.add_argument('--minimum_event_length',
                        help="Minimum event length",
                        type=float)

    parser.add_argument('--minimum_event_gap',
                        help="Minimum event gap",
                        type=float)

    parser.add_argument('--publication',
                        help="Strip visual elements out, use to generate figures for publication",
                        action="store_true")

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    
    parser.add_argument('-sp',
                        dest='save_path',
                        default=None,
                        type=str,
                        help="Save the figure at the given path without opening a figure window (useful for batch "
                             "processing of the figures",
                        action='store',
                        required=False)
    
    return vars(parser.parse_args(argv[1:]))


def main(argv):
    """
    """
    ui = dcase_util.ui.FancyPrinter()
    ui.section_header('sed_visualizer')
    parameters = process_arguments(argv)

    if parameters['spectrogram']:
        mode = 'spectrogram'

    elif parameters['time_domain']:
        mode = 'time_domain'

    else:
        mode = None

    if parameters['events']:
        active_events = parameters['events']
    else:
        active_events = None

    if parameters['publication']:
        publication_mode = True

    else:
        publication_mode = False

    audio_container = dcase_util.containers.AudioContainer().load(
        parameters['audio_file']
    )
    ui.data(field='Audio file', value=parameters['audio_file'])

    event_lists = {}
    event_list_order = []

    ui.line('Event lists', indent=2)
    ui.row('ID', 'Label', 'Event list file',
           widths=[5, 15, 40],
           types=['int', 'str15', 'str40'],
           indent=4
           )
    ui.row('-', '-', '-')
    for id, list_file in enumerate(parameters['list']):
        ui.row(id, parameters['names'][id], list_file)

        event_lists[parameters['names'][id]] = dcase_util.containers.MetaDataContainer().load(list_file)
        event_list_order.append(parameters['names'][id])

    ui.line()
    ui.data(field='Mode', value=mode)
    ui.data(field='Active events', value=active_events)
    ui.data(field='Publication mode', value=publication_mode)
    ui.data(field='minimum event length', value=parameters['minimum_event_length'], unit='sec')
    ui.data(field='minimum event gap', value=parameters['minimum_event_gap'], unit='sec')
    ui.sep()

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        event_list_order=event_list_order,
        active_events=active_events,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        mode=mode,
        minimum_event_length=parameters['minimum_event_length'],
        minimum_event_gap=parameters['minimum_event_gap'],
        publication_mode=publication_mode
    )
    
    if parameters['save_path'] is not None:
        vis.save(parameters['save_path'])
    else:
        vis.show()
    

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)
