#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Event list handling
"""

__all__ = ['EventList',
           'unique_event_labels',
           'max_event_offset']

import copy

class EventList(list):
    def __init__(self, *args):
        """Constructor

        This class is inherited from list class.

        Returns
        -------
            Nothing

        """
        list.__init__(self, *args)

    @property
    def valid_event_list(self):
        if 'event_label' in self[0] and 'event_onset' in self[0] and 'event_offset' in self[0]:
            return True
        else:
            return False

    @property
    def event_count(self):
        """Get number of events

        Returns
        -------
        event_count: integer > 0

        """

        if self.valid_event_list:
            return len(self)

    @property
    def event_label_count(self):
        """Get number of unique event labels

        Returns
        -------
        event_label_count: float > 0

        """

        if self.valid_event_list:
            return len(self.unique_event_labels)

    @property
    def unique_event_labels(self):
        """Get unique event labels

        Returns
        -------
        labels: list, shape=(n,)
            Unique labels in alphabetical order

        """

        if self.valid_event_list:
            return unique_event_labels(self)

    @property
    def max_event_offset(self):
        """Find the offset (end-time) of last event

        Returns
        -------
        max_offset: float > 0
            maximum offset

        """

        if self.valid_event_list:
            return max_event_offset(self)

    def filter(self, minimum_event_length=None, minimum_event_gap=None):
        """Filter event list. Makes sure that minimum event length and minimum event gap conditions are met per event label class.

        Parameters
        ----------
        minimum_event_length : float > 0.0
            Minimum event length in seconds, shorten than given are filtered out from the output.
            (Default value=None)

        minimum_event_gap : float > 0.0
            Minimum allowed gap between events in seconds from same event label class.
            (Default value=None)

        Returns
        -------
        nothing

        """

        if self.valid_event_list:
            return EventList(filter_event_list(event_list=self, minimum_event_length=minimum_event_length, minimum_event_gap=minimum_event_gap))


def unique_event_labels(event_list):
    """Find the unique event labels

    Parameters
    ----------
    event_list : list, shape=(n,)
        A list containing event dicts

    Returns
    -------
    labels: list, shape=(n,)
        Unique labels in alphabetical order

    """

    labels = []
    for event in event_list:
        if 'event_label' in event and event['event_label'] not in labels:
            labels.append(event['event_label'])

    labels.sort()
    return labels


def max_event_offset(event_list):
    """Find the offset (end-time) of last event

    Parameters
    ----------
    event_list : list, shape=(n,)
        A list containing event dicts

    Returns
    -------
    max_offset: float > 0
        maximum offset

    """

    max_offset = 0
    for event in event_list:
        if event['event_offset'] > max_offset:
            max_offset = event['event_offset']
    return max_offset


def filter_event_list(event_list, minimum_event_length=None, minimum_event_gap=None):
    """Filter event list. Makes sure that minimum event length and minimum event gap conditions are met per event label class.

    Parameters
    ----------
    event_list : list, shape=(n,)
        A list containing event dicts

    minimum_event_length : float > 0.0
        Minimum event length in seconds, shorten than given are filtered out from the output.
        (Default value=None)

    minimum_event_gap : float > 0.0
        Minimum allowed gap between events in seconds from same event label class.
        (Default value=None)

    Returns
    -------
    event_list_filtered : list, shape=(n,)
        A list containing event dicts

    """

    event_labels = unique_event_labels(event_list)
    event_list_filtered = []
    for event_label in event_labels:
        events_items = []
        for event in event_list:
            if event['event_label'] == event_label:
                events_items.append(event)

        # Sort events
        events_items = sorted(events_items, key=lambda k: k['event_onset'])

        # 1. remove short events
        event_results_1 = []
        for event in events_items:
            if minimum_event_length is not None:
                if event['event_offset']-event['event_onset'] >= minimum_event_length:
                    event_results_1.append(event)
            else:
                event_results_1.append(event)

        if len(event_results_1) and minimum_event_gap is not None:
            # 2. remove small gaps between events
            event_results_2 = []

            # Load first event into event buffer
            buffered_event_onset = event_results_1[0]['event_onset']
            buffered_event_offset = event_results_1[0]['event_offset']
            for i in range(1, len(event_results_1)):
                if event_results_1[i]['event_onset'] - buffered_event_offset > minimum_event_gap:
                    # The gap between current event and the buffered is bigger than minimum event gap,
                    # store event, and replace buffered event
                    current_event = copy.copy(event_results_1[i])
                    current_event['event_onset'] = buffered_event_onset
                    current_event['event_offset'] = buffered_event_offset
                    event_results_2.append(current_event)

                    buffered_event_onset = event_results_1[i]['event_onset']
                    buffered_event_offset = event_results_1[i]['event_offset']
                else:
                    # The gap between current event and the buffered is smaller than minimum event gap,
                    # extend the buffered event until the current offset
                    buffered_event_offset = event_results_1[i]['event_offset']

            # Store last event from buffer
            current_event = copy.copy(event_results_1[len(event_results_1)-1])
            current_event['event_onset'] = buffered_event_onset
            current_event['event_offset'] = buffered_event_offset
            event_results_2.append(current_event)

            event_list_filtered += event_results_2
        else:
            event_list_filtered += event_results_1

    return event_list_filtered