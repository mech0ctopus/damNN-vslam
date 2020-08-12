#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH --mem 16G
#SBATCH -t 2:00:00

module load python/gcc-8.2.0/3.7.6
module load cuda10.1/toolkit/10.1.105
module load cuda10.1/blas/10.1.105
module load eigen/gcc-8.2.0/3.3.7

source /home/cdmiller/damnn/bin/activate

#python3 /home/cdmiller/damNN-vslam/models/models.py
python3 /home/cdmiller/damNN-vslam/damnn_vslam_vo_only.py
