# !/bin/bash
set -x
cd autoregressive
# add pwd to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`

torchrun \
--nnodes=1 --nproc_per_node=1 \
--master_port=12345 \
train/train_c2i.py --no-compile --log-every 10 --code-path /mnt/ceph_rbd/zbc/slot_ar/codes_with_labels.pth \
--cloud-save-path /mnt/ceph_rbd/zbc/slot_ar/output/autoregressive/GPT-B-Woof-16384 \
--dataset imagewoof_code \
--results-dir /mnt/ceph_rbd/zbc/slot_ar/output/autoregressive/GPT-B-Woof-16384/res