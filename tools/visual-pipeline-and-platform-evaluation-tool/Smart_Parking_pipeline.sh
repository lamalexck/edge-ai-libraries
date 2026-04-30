#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

VIDEO=${1:-/videos/input/metro_smart_parking.mp4}
MAX_CHANNELS=${MAX_CHANNELS:-10}
RESULT_FILE=${RESULT_FILE:-/output/SmartParking_results.txt}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_RESOLVER="${MODEL_RESOLVER:-${SCRIPT_DIR}/resolve_model_artifact.sh}"

resolve_model_artifact() {
    local model_ref="$1"
    local artifact_type="$2"

    if [ -z "$model_ref" ]; then
        echo ""
        return 0
    fi

    if [ ! -x "$MODEL_RESOLVER" ]; then
        echo "$model_ref"
        return 0
    fi

    "$MODEL_RESOLVER" "$model_ref" "$artifact_type"
}

channel() {
    local c=${1:-1}

    if [ "$VARIANT" == "CPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=opencv
            device=CPU
            threshold=0.7
            inference-interval=1
            inference-region=full-frame
            model-instance-id=inst0-cpu-${c} !
        queue !
        gvaclassify
            model=${MODEL2}
            pre-process-backend=opencv
            device=CPU
            inference-interval=1
            model-instance-id=inst1-cpu-${c}
            inference-region=roi-list !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=8
            inference-region=full-frame
            inference-interval=1
            nireq=2
            threshold=0.7
            model-instance-id=instgpu0-${c} !
        queue !
        gvaclassify
            batch-size=8
            model=${MODEL2}
            pre-process-backend=va-surface-sharing
            device=GPU
            inference-interval=1
            inference-region=roi-list
            nireq=2
            model-instance-id=instgpu1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=1
            inference-region=full-frame
            nireq=2
            threshold=0.7
            model-instance-id=instnpu0-${c} !
        queue !
        gvaclassify
            model=${MODEL2}
            pre-process-backend=va
            device=NPU
            inference-interval=1
            inference-region=roi-list
            batch-size=1
            nireq=2
            model-instance-id=instnpu1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=8
            inference-region=full-frame
            inference-interval=1
            nireq=2
            threshold=0.7
            model-instance-id=instgpu-npu0-${c} !
        queue !
        gvaclassify
            model=${MODEL2}
            pre-process-backend=va
            device=NPU
            inference-interval=1
            inference-region=roi-list
            batch-size=1
            nireq=2
            model-instance-id=instgpu-npu1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-region=full-frame
            inference-interval=3
            nireq=0
            threshold=0.7
            model-instance-id=instgpu-opt0-${c} !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !        
        gvaclassify
            batch-size=0
            model=${MODEL2}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            inference-interval=3
            inference-region=roi-list
            nireq=0
            model-instance-id=instgpu-opt1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=3
            inference-region=full-frame
            nireq=2
            threshold=0.7
            model-instance-id=instnpu-opt0-${c} !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue ! 
        gvaclassify
            model=${MODEL2}
            pre-process-backend=va
            device=NPU
            inference-interval=3
            inference-region=roi-list
            batch-size=1
            nireq=2
            model-instance-id=instnpu-opt1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-region=full-frame
            inference-interval=3
            nireq=0
            threshold=0.7
            model-instance-id=instgpu-npu-opt0-${c} !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue ! 
        gvaclassify
            model=${MODEL2}
            pre-process-backend=va
            device=NPU
            inference-interval=3
            inference-region=roi-list
            batch-size=1
            nireq=2
            model-instance-id=instgpu-npu-opt1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU_GPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            model=${MODEL1}
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-region=full-frame
            inference-interval=3
            nireq=2
            threshold=0.7
            model-instance-id=instnpu-gpu-opt0-${c} !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue ! 
        gvaclassify
            model=${MODEL2}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            inference-interval=3
            inference-region=roi-list
            batch-size=0
            nireq=0
            model-instance-id=instnpu-gpu-opt1-${c} !
        queue !
        gvawatermark !
        gvametaconvert add-empty-results=true !
        queue !
        gvafpscounter starting-frame=100 !
        fakesink name=default_output_sink_${c} "
    fi
}

mkdir -p /output

MODEL1_REF="yolo11s"
MODEL1="$(resolve_model_artifact "$MODEL1_REF" xml)"
MODEL2_REF="colorcls2"
MODEL2="$(resolve_model_artifact "$MODEL2_REF" xml)"

#for variant in "GPU" "NPU" "GPU_NPU" "GPU_Opt" "NPU_Opt" "GPU_NPU_Opt" "NPU_GPU_Opt"; do
#for variant in "GPU" "GPU_Opt" "NPU_Opt" "GPU_NPU_Opt" "NPU_GPU_Opt"; do
for variant in "NPU_GPU_Opt"; do

    VARIANT=${variant}

    pipeline=""
    for c in $(seq 1 ${MAX_CHANNELS}); do
        pipeline+="$(channel "${c}")"
    done

    $(python3 ./npu-monitor-tool.py -i 1000 --csv) &
    pid=$!

    echo "========================================" | tee -a "${RESULT_FILE}"
    echo "Pipeline for ${VARIANT} variant of ${MODEL1_REF} + ${MODEL2_REF}:" | tee -a "${RESULT_FILE}"
    gst-launch-1.0 -e ${pipeline} | grep "overall" | grep "number-streams=${MAX_CHANNELS}" | tee -a "${RESULT_FILE}"
    
    kill -s SIGINT "${pid}"
    echo "" | tee -a "${RESULT_FILE}"

    if [ -d npu_output ]; then
        mv npu_output "/output/SmartParking_npu_output_${variant}_${MODEL1_REF}_${MODEL2_REF}"
    fi

    sleep 5
done
