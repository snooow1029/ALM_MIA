#!/bin/bash

# Script to run conditional_mia_analysis.py for all dataset and prompt combinations
# Automatically finds MIA results files from both mf3 and af3 directories (original, noise, silence, tts, etc.)
# Usage: bash run_all_conditional_mia.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/conditional_mia_analysis.py"
MIA_RESULTS_BASE="${AUDIO_MIA_RESULTS_ROOT:-$(dirname "$SCRIPT_DIR")/two_stage_mia_results}"

# Log file
LOG_FILE="${SCRIPT_DIR}/run_all_conditional_mia.log"
echo "Starting batch run at $(date)" | tee -a "${LOG_FILE}"

# Counter for tracking progress
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_JOBS=()
FOUND_FILES=()

# Find all MIA results files (mf3 and af3)
echo "Scanning for MIA results files..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Iterate through all dataset directories
for dataset_dir in "${MIA_RESULTS_BASE}"/*; do
    # Check if it's a directory
    if [ ! -d "${dataset_dir}" ]; then
        continue
    fi
    
    dataset=$(basename "${dataset_dir}")
    
    # Skip non-dataset directories (like 0_1, 1_0, etc.)
    # You can add more filters here if needed
    if [[ "${dataset}" =~ ^[0-9_]+$ ]]; then
        continue
    fi
    
    # Look for prompt directories
    for prompt_dir in "${dataset_dir}"/prompt_*; do
        # Check if it's a directory
        if [ ! -d "${prompt_dir}" ]; then
            continue
        fi
        
        # Extract prompt letter from directory name (e.g., "prompt_a" -> "a")
        prompt=$(basename "${prompt_dir}" | sed 's/prompt_//')
        
        # Check for mf3 directory
        mf3_dir="${prompt_dir}/mf3"
        if [ -d "${mf3_dir}" ]; then
            # Look for modal directories under mf3
            for modal_dir in "${mf3_dir}"/*; do
                # Check if it's a directory
                if [ ! -d "${modal_dir}" ]; then
                    continue
                fi
                
                modal=$(basename "${modal_dir}")
                
                # Construct path to results file
                mia_results_path="${modal_dir}/two_stage_audio_mia_results_${modal}.json"
                
                # Check if the file exists
                if [ -f "${mia_results_path}" ]; then
                    FOUND_FILES+=("${dataset}|${prompt}|${modal}|${mia_results_path}")
                    echo "  Found: ${dataset}/prompt_${prompt}/mf3/${modal}/two_stage_audio_mia_results_${modal}.json" | tee -a "${LOG_FILE}"
                fi
            done
        fi
        
        # Also check for modal directories directly under prompt_dir (af3 results)
        for modal_dir in "${prompt_dir}"/*; do
            # Skip if it's the mf3 directory itself
            if [ "$(basename "${modal_dir}")" = "mf3" ]; then
                continue
            fi
            
            # Check if it's a directory
            if [ ! -d "${modal_dir}" ]; then
                continue
            fi
            
            modal=$(basename "${modal_dir}")
            
            # Construct path to results file
            mia_results_path="${modal_dir}/two_stage_audio_mia_results_${modal}.json"
            
            # Check if the file exists
            if [ -f "${mia_results_path}" ]; then
                FOUND_FILES+=("${dataset}|${prompt}|${modal}|${mia_results_path}")
                echo "  Found: ${dataset}/prompt_${prompt}/${modal}/two_stage_audio_mia_results_${modal}.json" | tee -a "${LOG_FILE}"
            fi
        done
    done
done

TOTAL_COMBINATIONS=${#FOUND_FILES[@]}
echo "" | tee -a "${LOG_FILE}"
echo "Total MIA files found: ${TOTAL_COMBINATIONS}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Iterate through found files
CURRENT_COUNT=0
for file_info in "${FOUND_FILES[@]}"; do
    CURRENT_COUNT=$((CURRENT_COUNT + 1))
    
    # Parse the file info (dataset|prompt|modal|path)
    IFS='|' read -r dataset prompt modal mia_results_path <<< "${file_info}"
    
    echo "[${CURRENT_COUNT}/${TOTAL_COMBINATIONS}] Running: ${dataset} prompt_${prompt} modal_${modal}" | tee -a "${LOG_FILE}"
    echo "  MIA results: ${mia_results_path}" | tee -a "${LOG_FILE}"
    
    # Run the analysis (output to both terminal and log file)
    # Use PIPESTATUS to capture the actual exit code of python command
    # The Python script will check if this combination already exists and skip if needed
    # Modal will be auto-detected from the path
    python "${PYTHON_SCRIPT}" \
        --dataset_name "${dataset}" \
        --mia_results_path "${mia_results_path}" \
        2>&1 | tee -a "${LOG_FILE}"
    
    PYTHON_EXIT_CODE=${PIPESTATUS[0]}
    if [ ${PYTHON_EXIT_CODE} -eq 0 ]; then
        echo "  ✓ SUCCESS: ${dataset} prompt_${prompt} modal_${modal}" | tee -a "${LOG_FILE}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ✗ FAILED: ${dataset} prompt_${prompt} modal_${modal} (exit code: ${PYTHON_EXIT_CODE})" | tee -a "${LOG_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_JOBS+=("${dataset}/prompt_${prompt}/modal_${modal}")
    fi
    
    echo "" | tee -a "${LOG_FILE}"
done

# Print summary
echo "========================================" | tee -a "${LOG_FILE}"
echo "Batch run completed at $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Total combinations: ${TOTAL_COMBINATIONS}" | tee -a "${LOG_FILE}"
echo "Successful: ${SUCCESS_COUNT}" | tee -a "${LOG_FILE}"
echo "Failed: ${FAILED_COUNT}" | tee -a "${LOG_FILE}"

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "Failed jobs:" | tee -a "${LOG_FILE}"
    for job in "${FAILED_JOBS[@]}"; do
        echo "  - ${job}" | tee -a "${LOG_FILE}"
    done
fi

echo "" | tee -a "${LOG_FILE}"
echo "Full log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
