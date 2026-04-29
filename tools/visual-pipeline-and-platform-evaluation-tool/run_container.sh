#!/bin/bash

CONTAINER=${1:-"intel/dlstreamer:2026.1.0-20260414-weekly-ubuntu24"}

docker run -ti --rm \
    --device /dev/dri \
    --device /dev/accel \
    --group-add 992 \
    --user 0:0 \
    -v /sys/kernel/debug:/sys/kernel/debug:ro \
    -v $(pwd)/SNVR_pipeline.sh:/home/dlstreamer/SNVR_pipeline.sh \
    -v $(pwd)/Goods_Detection_Classification_pipeline.sh:/home/dlstreamer/Goods_Detection_Classification_pipeline.sh \
    -v $(pwd)/Smart_Parking_pipeline.sh:/home/dlstreamer/Smart_Parking_pipeline.sh \
    -v $(pwd)/resolve_model_artifact.sh:/home/dlstreamer/resolve_model_artifact.sh \
    -v $(pwd)/../npu-monitor-tool/npu-monitor-tool.py:/home/dlstreamer/npu-monitor-tool.py \
    -v $(pwd)/shared/videos:/videos:ro \
    -v $(pwd)/shared/models:/models:ro \
    -v $(pwd)/shared/output:/output \
    ${CONTAINER}
    