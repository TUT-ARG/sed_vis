#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities
==================

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

__all__ = [_ for _ in dir() if not _.startswith('_')]