#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -u

MODELS_ROOT="${MODELS_ROOT:-/models/output}"
ARTIFACT_TYPE="${2:-xml}"
MODEL_REF="${1:-}"

declare -A MODEL_XML_PATHS=(
    [colorcls2]="public/colorcls2/FP32/colorcls2.xml"
    [efficientnet-b0_INT8]="pipeline-zoo-models/efficientnet-b0_INT8/FP16-INT8/efficientnet-b0.xml"
    [mobilenet-v2-pytorch]="omz/mobilenet-v2-pytorch/FP16/mobilenet-v2-pytorch.xml"
    [resnet-50-tf_INT8]="pipeline-zoo-models/resnet-50-tf_INT8/resnet-50-tf_i8.xml"
    [yolo11n]="public/yolo11n/INT8/yolo11n.xml"
    [yolo11s]="public/yolo11s/INT8/yolo11s.xml"
)

declare -A MODEL_JSON_PATHS=(
    [efficientnet-b0_INT8]="pipeline-zoo-models/efficientnet-b0_INT8/efficientnet-b0.json"
    [mobilenet-v2-pytorch]="omz/mobilenet-v2-pytorch/mobilenet-v2.json"
    [resnet-50-tf_INT8]="pipeline-zoo-models/resnet-50-tf_INT8/resnet-50-tf_i8.json"
)

print_usage() {
    echo "Usage: $0 <model-name-or-path> [xml|json]" >&2
}

resolve_from_static_map() {
    local model_name="$1"
    local artifact_extension="$2"
    local relative_path=""

    if [ "$artifact_extension" = "xml" ]; then
        relative_path="${MODEL_XML_PATHS[$model_name]:-}"
    else
        relative_path="${MODEL_JSON_PATHS[$model_name]:-}"
    fi

    if [ -z "$relative_path" ]; then
        return 1
    fi

    echo "${MODELS_ROOT}/${relative_path}"
}

if [ -z "$MODEL_REF" ]; then
    print_usage
    exit 1
fi

if [ "$ARTIFACT_TYPE" != "xml" ] && [ "$ARTIFACT_TYPE" != "json" ]; then
    print_usage
    exit 2
fi

if [[ "$MODEL_REF" == */* ]] || [[ "$MODEL_REF" == *.xml ]] || [[ "$MODEL_REF" == *.json ]]; then
    if [[ "$MODEL_REF" == *.${ARTIFACT_TYPE} ]]; then
        echo "$MODEL_REF"
        exit 0
    fi

    artifact_dir="$(dirname "$MODEL_REF")"
    artifact_basename="$(basename "$MODEL_REF")"
    artifact_stem="${artifact_basename%.*}"

    candidate_artifact="${artifact_dir}/${artifact_stem}.${ARTIFACT_TYPE}"
    if [ -f "$candidate_artifact" ] || [ ! -d "$artifact_dir" ]; then
        echo "$candidate_artifact"
        exit 0
    fi

    exit 0
fi

resolved_artifact="$(resolve_from_static_map "$MODEL_REF" "$ARTIFACT_TYPE")"

if [ -z "$resolved_artifact" ]; then
    if [ "$ARTIFACT_TYPE" = "json" ]; then
        echo ""
        exit 0
    fi
    echo "Error: no static .${ARTIFACT_TYPE} mapping found for model '${MODEL_REF}'." >&2
    exit 4
fi

echo "$resolved_artifact"