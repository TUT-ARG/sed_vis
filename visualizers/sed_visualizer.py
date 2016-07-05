#!/usr/bin/env python
"""
Visualizer for sound event detection system
"""

import sys
import os
import argparse
import textwrap
import sed_vis

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
    return vars(parser.parse_args(argv[1:]))


def main(argv):
    """
    """

    parameters = process_arguments(argv)
    if os.path.isfile(parameters['audio_file']):
        audio, fs = sed_vis.io.load_audio(parameters['audio_file'])
    else:
        raise IOError('Audio file not found ['+parameters['audio_file']+']')
    event_lists = {}
    event_list_order = []
    for id,list_file in enumerate(parameters['list']):
        print id, parameters['names'][id], list_file
        event_lists[parameters['names'][id]] = sed_vis.io.load_event_list(list_file)
        event_list_order.append(parameters['names'][id])

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

    vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                    event_list_order=event_list_order,
                                                    active_events=active_events,
                                                    audio_signal=audio,
                                                    sampling_rate=fs,
                                                    mode=mode,
                                                    minimum_event_length=parameters['minimum_event_length'],
                                                    minimum_event_gap=parameters['minimum_event_gap'],
                                                    publication_mode=publication_mode
                                                    )
    vis.show()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
