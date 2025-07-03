#!/usr/bin/env bash
set -x

GPUS=${GPUS:-8}
PORT=${PORT:-29503}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this


# train
PRETRAIN=${PRETRAINED_WEIGHTS}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --dataset_file mevis --with_box_refine --binary --freeze_text_encoder \
--epochs 6 --lr_drop 3 5 \
--output_dir=${OUTPUT_DIR} ${PY_ARGS}


WORKDIR=$(pwd)
echo ${WORKDIR}
cd ${OUTPUT_DIR}/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
cd ${WORKDIR}
python3 merge_lora_weights_and_save_hf_model.py



# inference
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
VERSION=${OUTPUT_DIR}/evf
python3 inference_mevis.py --with_box_refine --binary --freeze_text_encoder --inference --precision="fp32" \
--output_dir=${OUTPUT_DIR} --resume=${VERSION} --version=${VERSION} ${PY_ARGS}
cd ${OUTPUT_DIR}
mkdir Annotations
mv valid/* Annotations/
zip -q -r submission.zip Annotations
echo "Working path is: ${OUTPUT_DIR}"

