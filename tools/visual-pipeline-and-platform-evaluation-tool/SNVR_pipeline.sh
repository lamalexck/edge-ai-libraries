#!/bin/bash

VIDEO=${1:-/videos/input/TownCentreXVID.1min.mp4}
WIDTH=${2:-320}
HEIGHT=${3:-240}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_RESOLVER="${MODEL_RESOLVER:-${SCRIPT_DIR}/resolve_model_artifact.sh}"

#LABELS="/home/dlstreamer/dlstreamer/samples/labels/imagenet_2012.txt"

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

    if [ "$VARIANT" == "GPU" ]; then
        echo "
        filesrc location=${VIDEO} ! 
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        vapostproc !
        video/x-raw\(memory:VAMemory\) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0
            pre-process-backend=va
            device=NPU
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-gpu-npu
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-gpu-npu
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU_GPU" ]; then
        #   Just swapping NPU and GPU devices in the pipeline to see if it has any effect on performance
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-npu-gpu
            pre-process-backend=va-surface-sharing
            device=NPU
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-npu-gpu
            pre-process-backend=va
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        vapostproc !
        video/x-raw\(memory:VAMemory\) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=3
            inference-region=full-frame
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "GPU_NPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-gpu-npu
            pre-process-backend=va-surface-sharing
            device=GPU
            ie-config=GPU_THROUGHPUT_STREAMS=2
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-gpu-npu
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "    
    elif [ "$VARIANT" == "NPU_GPU_Opt" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_${c}.mp4
        t0_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-npu-gpu
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=3
            inference-region=full-frame
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-npu-gpu
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_${c} "
    elif [ "$VARIANT" == "NPU_P_GPU" ]; then
        echo "
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_npu_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_npu_${c}.mp4
        t0_npu_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-npu
            pre-process-backend=va
            device=NPU
            batch-size=1
            inference-interval=3
            inference-region=full-frame
            nireq=2 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-npu
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_npu_${c} 
        filesrc location=${VIDEO} !
        parsebin !
        tee name=t0_gpu_${c} !
        queue !
        splitmuxsink location=/tmp/SNVR-output_gpu_${c}.mp4
        t0_gpu_${c}. !
        queue !
        vah264dec !
        video/x-raw(memory:VAMemory) !
        gvafpscounter starting-frame=100 !
        gvadetect
            model=${MODEL1} ${model1_proc}
            model-instance-id=detect0-gpu
            pre-process-backend=va-surface-sharing
            device=GPU
            batch-size=0
            inference-interval=3
            inference-region=full-frame
            nireq=0 !
        queue !
        gvatrack tracking-type=short-term-imageless !
        queue !
        gvaclassify
            model=${MODEL2} ${model2_proc}
            model-instance-id=classify0-gpu
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
        vapostproc !
        video/x-raw(memory:VAMemory),width=${WIDTH},height=${HEIGHT} !
        fakesink name=default_output_sink_gpu_${c} "
    fi
}

RESULT_FILE="/output/SNVR_results.txt"
#for m1 in "yolo11s" "yolo11n"; do
for m1 in "yolo11s"; do
    MODEL1_REF=${m1}
    MODEL1="$(resolve_model_artifact "$MODEL1_REF" xml)"
    MODEL1_PROC="$(resolve_model_artifact "$MODEL1_REF" json 2>/dev/null)"

#    for m2 in "efficientnet-b0_INT8" "mobilenet-v2-pytorch" "resnet-50-tf_INT8"  ; do
    for m2 in "mobilenet-v2-pytorch"  ; do

        MODEL2_REF=${m2}
        MODEL2="$(resolve_model_artifact "$MODEL2_REF" xml)"
        MODEL2_PROC="$(resolve_model_artifact "$MODEL2_REF" json 2>/dev/null)"

#        for variant in "GPU" "NPU" "GPU_NPU" "NPU_GPU" "NPU_Opt" "GPU_NPU_Opt" "NPU_GPU_Opt" "NPU_P_GPU"; do
        for variant in "NPU_Opt"; do

            VARIANT=${variant}
            if [ "$VARIANT" == "NPU_P_GPU" ]; then
                max_chan=5
            else
                max_chan=10
            fi
            pipeline=""
            for c in $(seq 1 ${max_chan}); do
                pipeline+="$(channel ${c})"
            done
            $(python3 ./npu-monitor-tool.py -i 1000 --csv) &
            pid=$!
            echo "========================================" | tee -a ${RESULT_FILE}
            echo | tee -a ${RESULT_FILE}   
            echo "Pipeline for ${VARIANT} variant of ${m1} + ${m2}:" | tee -a ${RESULT_FILE}
            echo "- - - - - - - - - - - - - - - - - - - - - -" | tee -a ${RESULT_FILE}
            echo "$(channel 0)" | tee -a ${RESULT_FILE}
            echo "- - - - - - - - - - - - - - - - - - - - - -" | tee -a ${RESULT_FILE}
            gst-launch-1.0 -e ${pipeline} | grep "overall" | grep "number-streams=10" | tee -a ${RESULT_FILE}
            kill -s SIGINT ${pid}
            echo | tee -a ${RESULT_FILE}
            mv npu_output /output/SNVR_npu_output_${variant}_${m1}_${m2}
            sleep 5
        done
    done
done