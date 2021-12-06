#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 
/bin/echo "##" $(whoami) is compiling AI Essentials Module1 -- DPPY kmeans kernel - 1 of 2 kmeans_kernel.py
mpirun -n 4 python lab/kmeans_spmd.py

