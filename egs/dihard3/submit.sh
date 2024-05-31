#!/bin/sh
#PBS -j oe
#PBS -m ae
#PBS -M zzz128@stu.pku.edu.cn
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=23:00:00
#PBS -q normal
#PBS -P personal-alibabaz
#PBS -N OSDC-dihard3

cd "$PBS_O_WORKDIR" || exit
module load miniforge3
conda activate asteroid
cd ~/zizheng/OSDC/egs/dihard3 || exit
./train.sh