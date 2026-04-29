#!/bin/bash

VIDEO=${1:-/videos/input/TownCentreXVID.1min.mp4}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_RESOLVER="${MODEL_RESOLVER:-${SCRIPT_DIR}/resolve_model_artifact.sh}"

LABELS="/home/dlstreamer/dlstreamer/samples/labels/imagenet_2012.txt"
#MODEL2_PROC_PATH="/home/dlstreamer/dlstreamer/samples/gstreamer/model_proc/public/preproc-aspect-ratio.json"

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

function channel() {
    c=${1:-1}

    if [ -z "$MODEL1_PROC" ]; then
        model1_proc=''
    else
        model1_proc="model-proc=${MODEL1_PROC}"
    fi

    if [ -z "$MODEL2_PROC" ]; then
        model2_proc=''
    else
        model2_proc="model-proc=${MODEL2_PROC}"
    fi

    if [ "$VARIANT" == "CPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            batch-size=1
            model-instance-id=detect0-cpu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=CPU
            pre-process-backend=opencv
            ie-config=CPU_THROUGHPUT_STREAMS=2
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=1
            model-instance-id=classify0-cpu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=CPU
            inference-region=roi-list
            pre-process-backend=opencv
            ie-config=CPU_THROUGHPUT_STREAMS=2
            nireq=0
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_sink_${c} sync=false async=false "
    elif [ "$VARIANT" == "GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            batch-size=0
            model-instance-id=detect0-gpu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=GPU
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=0
            model-instance-id=classify0-gpu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=GPU
            inference-region=roi-list
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_sink_${c} sync=false async=false "
    elif [ "$VARIANT" == "NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            batch-size=1
            model-instance-id=detect0-npu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=NPU
            pre-process-backend=va
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=1
            model-instance-id=classify0-npu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=NPU
            inference-region=roi-list
            pre-process-backend=va
            nireq=0
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_sink_${c} sync=false async=false "
    elif [ "$VARIANT" == "GPU_NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 !
        gvadetect
            batch-size=0
            model-instance-id=detect0-gpu-npu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=GPU
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=1
            model-instance-id=classify0-gpu-npu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=NPU
            inference-region=roi-list
            pre-process-backend=va
            nireq=2
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_sink_${c} sync=false async=false "
    elif [ "$VARIANT" == "NPU_GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 ! 
        gvadetect
            batch-size=1
            model-instance-id=detect0-npu-gpu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=NPU
            pre-process-backend=va
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=0
            model-instance-id=classify0-npu-gpu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=GPU
            inference-region=roi-list
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_sink_${c} sync=false async=false "
    elif [ "$VARIANT" == "NPU_P_GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        decodebin3 ! 
        gvadetect
            batch-size=1
            model-instance-id=detect0-npu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=NPU
            pre-process-backend=va
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=1
            model-instance-id=classify0-npu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=NPU
            inference-region=roi-list
            pre-process-backend=va
            nireq=2
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_gpu_sink_${c} sync=false async=false 
        filesrc location=${VIDEO} !
        decodebin3 ! 
        gvadetect
            batch-size=0
            model-instance-id=detect0-gpu
            model=${MODEL1} ${model1_proc}
            threshold=0.5
            inference-interval=3
            scale-method=fast
            device=GPU
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            batch-size=0
            model-instance-id=classify0-gpu
            labels=${LABELS}
            model=${MODEL2} ${model2_proc}
            device=GPU
            inference-region=roi-list
            pre-process-backend=va-surface-sharing
            ie-config=GPU_THROUGHPUT_STREAMS=2
            nireq=0
            inference-interval=3
            reclassify-interval=1 !
        gvafpscounter starting-frame=100 !
        gvawatermark !
        gvametaconvert !
        queue !
        gvametapublish file-format=json-lines file-path=/dev/null !
        fakesink name=default_output_npu_sink_${c} sync=false async=false"
    fi
}

RESULT_FILE="/output/GoodsDetectionClassification_results.txt"
#for m1 in "yolo11s" "yolo11n"; do
for m1 in "yolo11s"; do
    MODEL1_REF=${m1}
    MODEL1="$(resolve_model_artifact "$MODEL1_REF" xml)"
    MODEL1_PROC="$(resolve_model_artifact "$MODEL1_REF" json 2>/dev/null)"

#    for m2 in "mobilenet-v2-pytorch" "resnet-50-tf_INT8"  "efficientnet-b0_INT8"; do
    for m2 in "efficientnet-b0_INT8"; do
        MODEL2_REF=${m2}
        MODEL2="$(resolve_model_artifact "$MODEL2_REF" xml)"
        MODEL2_PROC="$(resolve_model_artifact "$MODEL2_REF" json 2>/dev/null)"

#        for variant in 'NPU_GPU' 'GPU_NPU' 'NPU_P_GPU' "GPU"; do # "GPU_NPU"; do
        for variant in 'GPU'; do # "GPU_NPU"; do
            VARIANT=${variant}

            pipeline=""
            if [ "$VARIANT" == "NPU_P_GPU" ]; then
                max_chan=5
            else
                max_chan=10
            fi
            for c in $(seq 1 ${max_chan}); do
                pipeline+="$(channel ${c})"
            done
            $(python3 ./npu-monitor-tool.py -i 1000 --csv) &
            pid=$!
            echo "========================================" | tee -a ${RESULT_FILE}
            echo | tee -a ${RESULT_FILE}            
            echo "Pipeline for ${VARIANT} variant of ${MODEL1_REF} + ${MODEL2_REF}:" | tee -a ${RESULT_FILE}
            echo "- - - - - - - - - - " | tee -a ${RESULT_FILE}
            echo $(channel 0) | tee -a ${RESULT_FILE}
            echo "- - - - - - - - - - " | tee -a ${RESULT_FILE}
            gst-launch-1.0 -e ${pipeline} | grep "overall" | grep "number-streams=10" | tee -a ${RESULT_FILE}
            kill -s SIGINT ${pid}
            echo | tee -a ${RESULT_FILE}            
            mv npu_output /output/GoodsDetectionClassification_npu_output_${variant}_${m1}_${m2}
        done
    done
done