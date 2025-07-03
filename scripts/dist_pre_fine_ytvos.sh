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
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py --dataset_file all --binary --with_box_refine --batch_size 8 --num_frames 1 --epochs 10 --lr 1e-5 --lr_drop 6 8 --output_dir ./work_dir/pretrain --ngpu=8

WORKDIR=$(pwd)
echo ${WORKDIR}
cd ./work_dir/pretrain/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
cd ${WORKDIR}
python3 merge_lora_weights_and_save_hf_model.py --weight ./work_dir/pretrain/pytorch_model.bin --save_path ./work_dir/pretrain/evf
# The save_path should be the same as the PRETRAINED_WEIGHTS.

# train
PRETRAIN=${PRETRAINED_WEIGHTS}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --with_box_refine --binary --freeze_text_encoder --get_pretrain \
--epochs 6 --lr_drop 3 5 \
--output_dir=${OUTPUT_DIR} --version=${PRETRAIN} ${PY_ARGS}


WORKDIR=$(pwd)
echo ${WORKDIR}
cd ${OUTPUT_DIR}/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
cd ${WORKDIR}
python3 merge_lora_weights_and_save_hf_model.py


# inference
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
VERSION=${OUTPUT_DIR}/evf
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --inference --precision="fp32" \
--output_dir=${OUTPUT_DIR} --resume=${VERSION} --version=${VERSION} ${PY_ARGS}
cd ${OUTPUT_DIR}
mkdir Annotations
mv valid/* Annotations/
zip -q -r submission.zip Annotations

echo "Working path is: ${OUTPUT_DIR}"

