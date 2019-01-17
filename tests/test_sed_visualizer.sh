#!/usr/bin/env bash

../visualizers/sed_visualizer.py -a ../tests/data/a001.wav -l ../tests/data/a001.ann ../tests/data/a001_full.ann ../tests/data/a001_system_output_prob.csv -n reference1 full reference2 -e "bird singing" "car passing by" --publication