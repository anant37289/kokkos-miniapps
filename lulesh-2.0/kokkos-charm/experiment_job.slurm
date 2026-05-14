#!/bin/bash

# Default values if not provided via sbatch export
if [ -z "$NP" ]; then echo "Error: NP (total GPUs) not set"; exit 1; fi
if [ -z "$BASE_S" ]; then echo "Error: BASE_S (global grid size) not set"; exit 1; fi
if [ -z "$I" ]; then echo "Error: I (iterations) not set"; exit 1; fi
if [ -z "$OUTPUT_FILE" ]; then echo "Error: OUTPUT_FILE not set"; exit 1; fi

RESULT_FILE="$OUTPUT_FILE"
NUM_RUNS=2

# Initialize result file header if it doesn't exist
if [ ! -f "$RESULT_FILE" ]; then
    echo "TotalGPUs,Nodes,Chares,ODF,PPN,S,Iterations,TotalTime(s),TimePerIter(us)" > "$RESULT_FILE"
fi

# Timeout (seconds) for each srun invocation and max retries before skipping
RUN_TIMEOUT=${RUN_TIMEOUT:-80}
MAX_RETRIES=${MAX_RETRIES:-2}

echo "Starting $NUM_RUNS runs for: NP=$NP Nodes=$SLURM_NNODES BASE_S=$BASE_S I=$I"
echo "Per-run timeout: ${RUN_TIMEOUT}s | Max retries: $MAX_RETRIES"

# Environment setup
# export LD_LIBRARY_PATH=/u/ajain18/oldCharm/multicore-linux-x86_64-cuda/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/u/ajain18/reconverse_charm_integration/lci_install/lib64:$LD_LIBRARY_PATH
# export LCI_ATTR_BACKEND=ofi
# export FI_CXI_RX_MATH_MODE=hybrid
# export OFI_ROOT=/opt/cray/libfabric/1.15.2.0
# export LCI_ROOT=/u/ajain18/reconverse_charm_integration/lci_install
# export FI_MR_CACHE_MONITOR=userfaultfd
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/ajain18/hapi_hip/shogo/charm/kokkos/install/lib64
# export PMI_MAX_KVS_ENTRIES=4096
export LD_LIBRARY_PATH=/u/ajain18/temp/charm/multicore-linux-x86_64-cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/u/ajain18/reconverse_charm_integration/lci_install_cache/lib64:$LD_LIBRARY_PATH
export LCI_ATTR_BACKEND=ofi
export FI_CXI_RX_MATH_MODE=hybrid
export LCI_ATTR_BACKEND=ofi
export FI_CXI_RX_MATH_MODE=hybrid
export OFI_ROOT=/opt/cray/libfabric/1.15.2.0
export LCI_ROOT=/u/ajain18/reconverse_charm_integration/lci_install_cache
export FI_MR_CACHE_MONITOR=userfaultfd
export LCI_NET_LOCK_MODE=poll
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/ajain18/hapi_hip/shogo/charm/kokkos/install/lib64
export PMI_MAX_KVS_ENTRIES=16384
if [ "$DRY_RUN" = false ]; then
    nvidia-cuda-mps-control -d
fi

read -r -a TARGET_ODF_LIST <<< "$ODF_LIST_STR"

declare -A ran_N_map

for target_odf in "${TARGET_ODF_LIST[@]}"; do
    # Calculate target N (Total Chares)
    target_n=$((NP * target_odf))

    # Determine the nearest perfect cube for N
    c=1
    while [ $((c*c*c)) -lt $target_n ]; do
        c=$((c+1))
    done
    if [ $c -gt 1 ]; then
        diff1=$(( target_n - (c-1)*(c-1)*(c-1) ))
        diff2=$(( c*c*c - target_n ))
        if [ $diff1 -lt $diff2 ]; then
            c=$((c-1))
        fi
    fi

    N=$((c * c * c))

    # Avoid duplicate runs if different target_odfs collapse to the same cube N
    c_key="${NP}_${N}"
    if [[ "${ran_N_map[$c_key]}" == "1" ]]; then
        continue
    fi
    ran_N_map[$c_key]="1"

    actual_odf=$(awk "BEGIN {printf \"%.2f\", $N/$NP}")
    max_int_odf=$((N / NP))
    if (( $(echo "$actual_odf < 1.0" | bc -l) )); then
        continue
    fi
    if [ $max_int_odf -eq 0 ]; then max_int_odf=1; fi

    ppn_list=()
    p=$max_int_odf
    if [ "$p" -gt 8 ]; then
        p=8
    fi
    while true; do
        ppn_list+=( "$p" )
        if [ "$p" -eq 1 ]; then break; fi
        if [ $(( p % 2 )) -ne 0 ]; then break; fi
        p=$(( p / 2 ))
    done

    for ppn in "${ppn_list[@]}"; do
        chare_s=$((BASE_S / c))
        TOTAL_TIME_SUM=0
        TIME_PER_ITER_SUM=0
        
        CMD_RUN="srun -u --mpi=cray_shasta -n $((NP * ppn)) ./build/lulesh +pe $((NP * ppn)) -s $chare_s -i $I -n $N -b 0 -r 1 +gpushm +gpuipceventpool $((2048/ppn)) +gpucommbuffer $((2048/ppn))"

        echo "  Running: N=$N (Cube: ${c}^3), TargetODF=$target_odf, ActualODF=$actual_odf, PPN=$ppn, S_per_chare=$chare_s"


        if [ "$DRY_RUN" = true ]; then
            echo "    [DRY-RUN] $CMD_RUN > $OUTPUT_LOG"
            continue
        fi
        SUCCESSFUL_RUNS=0
        for i in $(seq 1 $NUM_RUNS); do
            OUTPUT_LOG="logs/run_${NP}_odf${target_odf}_ppn${ppn}_${SLURM_JOB_ID}_${i}.log"
            RUN_SUCCESS=false
            for attempt in $(seq 0 $MAX_RETRIES); do
                if [ $attempt -gt 0 ]; then
                    echo "  Retry $attempt/$MAX_RETRIES for run $i (N=$N ppn=$ppn)..."
                fi
                timeout $RUN_TIMEOUT $CMD_RUN > "$OUTPUT_LOG" 2>&1
                if [ $? -eq 124 ]; then
                    echo "  WARNING: run $i timed out after ${RUN_TIMEOUT}s (attempt $((attempt+1))/$((MAX_RETRIES+1)))"
                else
                    RUN_SUCCESS=true
                    break
                fi
            done
            if [ "$RUN_SUCCESS" = false ]; then
                echo "  SKIPPING run $i for N=$N ppn=$ppn after $((MAX_RETRIES+1)) attempts"
                continue
            fi

            T_TIME=$(awk '/Elapsed time/ {print $4}' "$OUTPUT_LOG")
            if [ -z "$T_TIME" ]; then
                echo "WARNING: Could not parse elapsed time from $OUTPUT_LOG"
                continue
            fi

            I_TIME=$(awk "BEGIN {printf \"%.2f\", ($T_TIME / $I) * 1000000}")
            TOTAL_TIME_SUM=$(echo "$TOTAL_TIME_SUM + $T_TIME" | bc)
            TIME_PER_ITER_SUM=$(echo "$TIME_PER_ITER_SUM + $I_TIME" | bc)
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        done

        if [ $SUCCESSFUL_RUNS -gt 0 ]; then
            AVG_TOTAL_TIME=$(echo "scale=3; $TOTAL_TIME_SUM / $SUCCESSFUL_RUNS" | bc)
            AVG_TIME_PER_ITER=$(echo "scale=3; $TIME_PER_ITER_SUM / $SUCCESSFUL_RUNS" | bc)
            echo "  Averages: Total=$AVG_TOTAL_TIME s, PerIter=$AVG_TIME_PER_ITER us"
            echo "$NP,$SLURM_NNODES,$N,$actual_odf,$ppn,$chare_s,$I,$AVG_TOTAL_TIME,$AVG_TIME_PER_ITER" >> "$RESULT_FILE"
        else
            echo "SKIPPED: No successful runs for N=$N ppn=$ppn — not writing to results"
        fi
    done
done

# MPI run
MPI_RESULT_FILE="MPI_${OUTPUT_FILE}"


# If Num GPUs is not perfect cube exit
mpi_c=1
while [ $((mpi_c * mpi_c * mpi_c)) -lt $NP ]; do
    mpi_c=$((mpi_c + 1))
done

if [ $((mpi_c * mpi_c * mpi_c)) -ne $NP ]; then
    exit 0
fi
S=$((BASE_S / mpi_c))
echo "MPI: NP=$NP, cube=${mpi_c}^3, S_per_rank=$S"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_DEBUG=1
export MPICH_OFI_VERBOSE=1
export FI_HMEM=cuda
export FI_CXI_ATS=0

TOTAL_TIME_SUM=0
TIME_PER_ITER_SUM=0 

if [ "$DRY_RUN" = true ]; then
    echo "    [DRY-RUN] MPI run with NP=$NP S=$S I=$I"
    exit 0
fi
CMD_MPI_RUN="srun --mpi=cray_shasta -n $NP /u/ajain18/lulesh/lulesh-2.0/kokkos-no-uvm/build/lulesh -s $S -i $I -b 0 -r 1"
for i in $(seq 1 $NUM_RUNS); do
    OUTPUT_LOG="logs/run_${NP}_${i}_MPI.log"
    echo "--- Run $i of $NUM_RUNS ---"
    
    $CMD_MPI_RUN > $OUTPUT_LOG   

    if [ $? -ne 0 ]; then
        echo "Error: srun failed on run $i"
        exit 1
    fi

    # Extract times using awk
    T_TIME=$(awk '/Elapsed time/ {print $4}' "$OUTPUT_LOG")
    
    # Check if we successfully parsed elapsed time
    if [ -z "$T_TIME" ]; then
        echo "Error: Could not parse elapsed time from $OUTPUT_LOG"
        exit 1
    fi
    
    # Per-iteration time in microseconds: (Elapsed_sec / Iters) * 1,000,000
    I_TIME=$(awk "BEGIN {printf \"%.2f\", ($T_TIME / $I) * 1000000}")

    # Accumulate sums using 'bc' for floating point addition
    TOTAL_TIME_SUM=$(echo "$TOTAL_TIME_SUM + $T_TIME" | bc)
    TIME_PER_ITER_SUM=$(echo "$TIME_PER_ITER_SUM + $I_TIME" | bc)
done

AVG_TOTAL_TIME=$(echo "scale=3; $TOTAL_TIME_SUM / $NUM_RUNS" | bc)
AVG_TIME_PER_ITER=$(echo "scale=3; $TIME_PER_ITER_SUM / $NUM_RUNS" | bc)
if [ ! -f "$MPI_RESULT_FILE" ]; then
    echo "TotalGPUs,Nodes,S,Iterations,TotalTime(s),TimePerIter(us)" > "$MPI_RESULT_FILE"
fi
echo "MPI Run Averages: Total=$AVG_TOTAL_TIME s, PerIter=$AVG_TIME_PER_ITER us"
echo "$NP,$SLURM_NNODES,$S,$I,$AVG_TOTAL_TIME,$AVG_TIME_PER_ITER" >> "$MPI_RESULT_FILE"
