#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI Essentials Module4 --  04_analyzeGalaxyBatch.py
python 04_analyzeGalaxyBatch.py

