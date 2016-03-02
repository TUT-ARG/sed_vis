#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities
==================


Event list operations
-----------

.. autosummary::
    :toctree: generated/

    event_list.unique_event_labels
    event_list.max_event_offset

Audio player
-----------

.. autosummary::
    :toctree: generated/

    audio_player.AudioPlayer
    audio_player.AudioPlayer.play
    audio_player.AudioPlayer.pause
    audio_player.AudioPlayer.stop
"""

from .audio_player import *
from .event_list import *

__all__ = [_ for _ in dir() if not _.startswith('_')]