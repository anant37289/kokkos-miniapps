#!/bin/bash

# Configuration
GPUS_PER_NODE=4
BASE_S=480
ITERATIONS=50
MAX_GPUS=32

# Parse arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -g, --gpus-per-node N   Number of GPUs per node (default: $GPUS_PER_NODE)"
    echo "  -m, --max-gpus N        Maximum total GPUs to scale to (default: $MAX_GPUS)"
    echo "  -s, --base-s N          Base chare grid size S (default: $BASE_S)"
    echo "  -i, --iters N           Number of iterations (default: $ITERATIONS)"
    echo "  --dry-run               Print commands without running them"
    exit 1
}

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--gpus-per-node)
        GPUS_PER_NODE="$2"
        shift; shift
        ;;
        -m|--max-gpus)
        MAX_GPUS="$2"
        shift; shift
        ;;
        -s|--base-s)
        BASE_S="$2"
        shift; shift
        ;;
        -i|--iters)
        ITERATIONS="$2"
        shift; shift
        ;;
        --dry-run)
        DRY_RUN=true
        shift
        ;;
        *)
        usage
        ;;
    esac
done

echo "Starting LULESH scaling experiments..."
echo "Base per-chare Size (S): $BASE_S"
echo "Iterations: $ITERATIONS"
echo "GPUs/Node: $GPUS_PER_NODE"
echo "Max GPUs: $MAX_GPUS"

# Target Overdecomposition Factors to approximate
TARGET_ODF_LIST=(1 2 4 8 16 32)

current_gpus=8
OUTPUT_FILE="lulesh_strong_scaling_benchmark_16GPU_check_for_block.csv"

while [ $current_gpus -le $MAX_GPUS ]; do
    
    # Calculate Node distribution
    if [ $current_gpus -lt $GPUS_PER_NODE ]; then
        NODES=1
        TASKS_PER_NODE=$current_gpus
        memory_request=$((current_gpus * 16))
    else
        NODES=$((current_gpus / GPUS_PER_NODE))
        TASKS_PER_NODE=$GPUS_PER_NODE
        memory_request=64
    fi

    echo "----------------------------------------------------------------"
    echo "Scale Step: $current_gpus GPUs ($NODES Nodes x $TASKS_PER_NODE Gpus/Node)"

    # Export variables for the sbatch script
    export NP=$current_gpus
    export BASE_S=$BASE_S
    export I=$ITERATIONS
    export ODF_LIST_STR="${TARGET_ODF_LIST[*]}"
    export DRY_RUN
    export OUTPUT_FILE

    CMD="sbatch --nodes=$NODES --job-name=LULESH_${current_gpus}_GPU --exclusive --account=mzu-delta-gpu --partition=gpuA40x4 --gpus-per-node=$TASKS_PER_NODE --time=00:30:00 experiment_job.slurm"

    if [ "$DRY_RUN" = true ]; then
        echo $CMD
        bash experiment_job.slurm
    else
        $CMD
    fi

    # Prepare for next step
    current_gpus=$((current_gpus * 2))
done
