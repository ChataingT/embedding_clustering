#!/bin/bash
#SBATCH --job-name=sweep_birch
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=shared-gpu
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9"
#SBATCH --array=0-1

# ── Environment ──────────────────────────────────────────────
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate

# ── Working directory ────────────────────────────────────────
cd /srv/beegfs/scratch/shares/schaerm/schaer2/video_sam2_pose/humanLISBET-paper

mkdir -p logs

# ── BIRCH threshold sweep ────────────────────────────────────
# Array index → threshold: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
# THRESHOLDS=(1.0 2.0 4.0 6.0)
THRESHOLDS=(2.8)
THR=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

EMBEDDINGS_DIR=/home/shares/schaerm/schaer2/thibaut/humanlisbet/output/full_train/embeddings/models/hlis-gr-w1200-o600-L8-H8-E128-FF512-auggeom-embedder/embeddings
OUTPUT_DIR=post_training/behavior_clustering/results/sweep_birch_thr_${THR}
CONFIG=post_training/behavior_clustering/configs/default.yaml

# ── Run with overridden threshold ────────────────────────────
# We create a temporary config with the modified BIRCH threshold.
TMP_CONFIG=$(mktemp /tmp/birch_sweep_XXXXXX.yaml)
sed "s/^  threshold: .*/  threshold: ${THR}/" "$CONFIG" > "$TMP_CONFIG"

python -m post_training.behavior_clustering.src.run_clustering \
    --embeddings-dir  "$EMBEDDINGS_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --config          "$TMP_CONFIG" \
    --log-level       INFO

rm -f "$TMP_CONFIG"

echo "SLURM array task $SLURM_ARRAY_TASK_ID (threshold=$THR) finished with exit code $?"
