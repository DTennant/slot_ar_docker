export CUDA_VISIBLE_DEVICES=2,3
# accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/vit_vqgan.yaml
accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/dit_vq_pretrained_imagenet.yaml