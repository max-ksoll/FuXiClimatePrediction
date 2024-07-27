#!/bin/bash
#SBATCH --job-name=FuXi_Training     # Jobname
#SBATCH --nodes=1                    # Anzahl der Knoten
#SBATCH --ntasks=1                   # Anzahl der Aufgaben
#SBATCH --cpus-per-task=20            # Anzahl der CPUs pro Aufgabe - funktioniert wohl nicht, mal schauen
#SBATCH --gres=gpu:1                 # Anzahl der GPUs pro Knoten
#SBATCH --time=04:00:00              # Laufzeit (hh:mm:ss)
#SBATCH --output=logs/%j.out         # Standard Output und Error Log
#SBATCH --error=logs/%j.err          # Error Log
#SBATCH --qos=acc_ehpc               # Quality of Service
#SBATCH --account=ehpc03             # Slurm Konto

# Laden der Module und Umgebungsvariablen
module load mkl intel impi hdf5 python cuda cudnn

source /gpfs/projects/ehpc03/max/venv/bin/activate

export DATA_PATH="/gpfs/projects/ehpc03/max/data"
export FIG_PATH="/gpfs/projects/ehpc03/max/data-viz"
export WANDB_DIR="/gpfs/projects/ehpc03/max"
export MODEL_DIR="/gpfs/projects/ehpc03/max/models"
export RAW_FC_LAYER=False
export SKIP_DATA_PREPARATION=true
export DEVICES=1
export NODES=1
export WANDB_MODE=offline

# Ausführen des Trainingsskripts
srun python /gpfs/projects/ehpc03/max/FuXiClimatePrediction/lightning_training_slurm.py
