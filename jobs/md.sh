#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N MD
#PBS -q gpu

module load go-1.19.4/apptainer-1.1.8
cd /work/smartire/OBIWAN_main
singularity exec --nv --bind $PWD ../OBIWAN_obi2/container/image.sif python3 MD_main.py > results/MD/logger_MD.txt 2>&1