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
#SBATCH --array=31-60%1

# Laden der Module und Umgebungsvariablen
module load mkl intel impi hdf5 python cuda cudnn

source /gpfs/projects/ehpc03/max/venv/bin/activate
export CARTOPY_DIR="/gpfs/projects/ehpc03/max/cartopy"
export MODULE_PATH="/gpfs/projects/ehpc03/max/FuXiClimatePrediction"

export MODEL_FILE="/gpfs/projects/ehpc03/max/models/epoch=531-step=74573.ckpt"
export DATA_PATH="/gpfs/projects/ehpc03/max/data/2006_2014.zarr"
export MEAN_DATA_PATH="/gpfs/projects/ehpc03/max/data/mean_2006_2014.zarr"
export OUTPUT_PATH="/gpfs/projects/ehpc03/max/metrics"
export BATCH_SIZE="8"

export AUTOREGRESSION_STEPS=$SLURM_ARRAY_TASK_ID

srun python /gpfs/projects/ehpc03/max/FuXiClimatePrediction/src/Eval/ModelEvaluatorMetrics.py
