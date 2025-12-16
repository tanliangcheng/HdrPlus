#!/bin/bash

BASE_DIR=/home/tanliangcheng/HdrPlus
RAW_DIR=${BASE_DIR}/data/hdr_plus_examples/RAWs
OUTPUT_DIR=${BASE_DIR}/outbmp

BURST_NAME=burst36
FRAME_NUM=8   # burst23_0 ~ burst23_7

cmake -B build 
cmake --build build -- -j8

# 1. Decode CR2
for ((i=0; i<FRAME_NUM; i++)); do
    ./build/ISPpipeline/DecodeCR2 \
        ${RAW_DIR}/${BURST_NAME}_${i}.CR2 \
        ${OUTPUT_DIR}/${BURST_NAME}_${i}.raw \
        ${OUTPUT_DIR}/${BURST_NAME}_${i}.txt
done

# 2. Run ISP pipeline
RAW_LIST=""
for ((i=0; i<FRAME_NUM; i++)); do
    RAW_LIST="${RAW_LIST} ${OUTPUT_DIR}/${BURST_NAME}_${i}.raw"
done

echo "Running ISP pipeline on burst: ${BURST_NAME} with frames: ${RAW_LIST}"

./build/ISPpipeline/testrawisp ${OUTPUT_DIR}/${BURST_NAME}_0.txt ${RAW_LIST}
