#!/bin/bash
#SBATCH --job-name=FuXi_Eval         # Jobname
#SBATCH --nodes=1                    # Anzahl der Knoten
#SBATCH --ntasks=1                   # Anzahl der Aufgaben
#SBATCH --cpus-per-task=20            # Anzahl der CPUs pro Aufgabe - funktioniert wohl nicht, mal schauen
#SBATCH --gres=gpu:1                 # Anzahl der GPUs pro Knoten
#SBATCH --time=12:00:00              # Laufzeit (hh:mm:ss)
#SBATCH --output=logs/%j.out         # Standard Output und Error Log
#SBATCH --error=logs/%j.err          # Error Log
#SBATCH --qos=acc_ehpc               # Quality of Service
#SBATCH --account=ehpc03             # Slurm Konto

# Laden der Module und Umgebungsvariablen
module load mkl intel impi hdf5 python cuda cudnn

source /gpfs/projects/ehpc03/max/venv/bin/activate

export DATA_PATH="/gpfs/projects/ehpc03/max/data"
export MODEL_FILE="/gpfs/projects/ehpc03/max/models/epoch=528-step=74814-v1.ckpt"
export CARTOPY_DIR="/gpfs/projects/ehpc03/max/cartopy"
export EVAL_START_YEAR="1958"
export AUTOREGRESSION_YEARS="10"
export OUTPUT_PATH="/gpfs/projects/ehpc03/max/videos"
export MODULE_PATH="/gpfs/projects/ehpc03/max/FuXiClimatePrediction/src"

# Ausführen des Trainingsskripts
srun python /gpfs/projects/ehpc03/max/FuXiClimatePrediction/src/Eval/ModelEvaluator.py
