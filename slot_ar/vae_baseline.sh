export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/vit_vqgan.yaml
accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/vae_slot_nested.yaml