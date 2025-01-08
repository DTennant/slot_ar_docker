export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --config_file=configs/a100_config.yaml train_net.py --cfg configs/example_imagenet.yaml
