#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

VIDEO=${1:-/videos/input/license-plate-detection.mp4}
MAX_CHANNELS=${MAX_CHANNELS:-10}
RESULT_FILE=${RESULT_FILE:-/output/LicensePlateRecognition_results.txt}

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
        gvafpscounter starting-frame=500 !
        gvadetect
            model=${MODEL1}
            model-instance-id=detect0_${c}
            device=CPU
            pre-process-backend=opencv
            batch-size=0
            inference-interval=3
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2}
            model-instance-id=classify0_${c}
            device=CPU
            pre-process-backend=opencv
            batch-size=0
            inference-interval=3
            nireq=0
            inference-region=roi-list
            reclassify-interval=1 !
        queue !
        gvawatermark !
        gvametaconvert format=json !
        gvametapublish method=file file-path=/dev/null !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvafpscounter starting-frame=500 !
        gvadetect
            model=${MODEL1}
            model-instance-id=detect0_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2}
            model-instance-id=classify0_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            nireq=0
            inference-region=roi-list
            reclassify-interval=1 !
        queue !
        gvawatermark !
        gvametaconvert format=json !
        gvametapublish method=file file-path=/dev/null !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvafpscounter starting-frame=500 !
        gvadetect
            model=${MODEL1}
            model-instance-id=detect0-gpu-npu_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2}
            model-instance-id=classify0-gpu-npu_${c}
            pre-process-backend=va
            device=NPU
            batch-size=0
            inference-interval=3
            nireq=0
            inference-region=roi-list
            reclassify-interval=1 !
        queue !
        gvawatermark !
        gvametaconvert format=json !
        gvametapublish method=file file-path=/dev/null !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvafpscounter starting-frame=500 !
        gvadetect
            model=${MODEL1}
            model-instance-id=detect0_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-interval=3
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2}
            model-instance-id=classify0_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-interval=3
            nireq=0
            inference-region=roi-list
            reclassify-interval=1 !
        queue !
        gvawatermark !
        gvametaconvert format=json !
        gvametapublish method=file file-path=/dev/null !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvafpscounter starting-frame=500 !
        gvadetect
            model=${MODEL1}
            model-instance-id=detect0-gpu-npu_${c}
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-interval=3
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2}
            model-instance-id=classify0-gpu-npu_${c}
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=3
            nireq=2
            inference-region=roi-list
            reclassify-interval=1 !
        queue !
        gvawatermark !
        gvametaconvert format=json !
        gvametapublish method=file file-path=/dev/null !
        fakesink name=default_output_sink_${c} "
    fi
}

mkdir -p /output

MODEL1_REF="/models/output/public/yolov8_license_plate_detector/FP32/yolov8_license_plate_detector.xml"
MODEL1="$(resolve_model_artifact "$MODEL1_REF" xml)"
MODEL2_REF="/models/output/public/ch_PP-OCRv4_rec_infer/FP32/ch_PP-OCRv4_rec_infer.xml"
MODEL2="$(resolve_model_artifact "$MODEL2_REF" xml)"

for variant in "GPU" "GPU_NPU" "GPU_Opt" "GPU_NPU_Opt"; do

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
        mv npu_output "/output/LicensePlateRecognition_npu_output_${variant}_yolov8_license_plate_detector_ch_PP-OCRv4_rec_infer"
    fi

    sleep 5
done
