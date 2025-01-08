# accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/vit_vqgan.yaml
accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/imagenet100_pretrainpixart.yaml
# python making_cache.py --cfg configs/imagenet1k_hybrid.yaml
# accelerate launch --config_file=configs/eidf_config.yaml train_net.py --cfg configs/imagenet1k_hybrid.yaml
