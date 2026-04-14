#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --output=logs/sweep_%A_%j.out
#SBATCH --error=logs/sweep_%A_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=shared-gpu
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"

# ── Environment ──────────────────────────────────────────────
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate

# ── Working directory ────────────────────────────────────────
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

mkdir -p logs


EMBEDDINGS_DIR=/home/shares/schaerm/schaer2/thibaut/humanlisbet/output/full_train/embeddings/models/hlis-gr-w1200-o600-L8-H8-E128-FF512-auggeom-embedder/embeddings
OUTPUT_DIR=post_training/behavior_clustering/results/v2
CONFIG=post_training/behavior_clustering/configs/v2.yaml

# ── Run with overridden threshold ────────────────────────────
# We create a temporary config with the modified BIRCH threshold.


python -m post_training.behavior_clustering.src.run_clustering \
    --embeddings-dir  "$EMBEDDINGS_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --config          "$CONFIG" \
    --log-level       INFO

echo "SLURM array task $SLURM_ARRAY_TASK_ID  finished with exit code $?"
