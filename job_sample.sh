#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N E_fr_spice_pretr
#PBS -q workq

module load singularity/3.10
cd /home/smartire/OBIWAN
singularity exec --nv --bind $PWD container/image.sif python3 training_routine.py \
                                                                --run_name "E_fr_spice_pretr" \
                                                                --model_name "obiwan" \
                                                                --batch_size_per_worker 256 \
                                                                --datasets "spice" \
                                                                --with_forces "false" \
                                                                --remove_toxic_molecules "false" \
                                                                --resume_training \
                                                                --freeze_first_layers \
                                                                > results/logger_E_fr_spice_pretr.txt 2>&1