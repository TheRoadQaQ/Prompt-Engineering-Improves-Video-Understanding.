###  TGIF for example
###  No prompt setting
experiment="0"

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json \
    --prefix ""

###  Fixed Prefix setting 
experiment="prefix0"

prefix="Watch this video carefully and try to answer the following question: "

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json \
    --prefix "${prefix}"

###  Auto prompt setting
experiment="auto0"

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video_auto_prefix.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/auto_test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json

### Semi Auto prompt setting
experiment="semi_auto0"

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video_routing.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/semi_auto_test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json

### Rewrite prompt setting
experiment="rewrite0"

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video_rewrite.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/rewrite_test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json 

### Video summary prompt setting
experiment="video_summary0"

CKPT_NAME=LLaVA-NeXT-Video-7B-DPO

GPU="0"

EVAL_DATA_DIR=...
OUTPUT_DIR=...

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python ./llava_next_video/llava_next_video_video_summary.py \
    --model-path LLaVA-NeXT-Video-7B-DPO-hf \
    --video-folder ${EVAL_DATA_DIR}/mp4 \
    --question-file ${EVAL_DATA_DIR}/test_q.json \
    --answer-file ${EVAL_DATA_DIR}/test_a.json \
    --output-file ${OUTPUT_DIR}/${CKPT_NAME}/${experiment}.json \