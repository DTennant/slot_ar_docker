export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file=configs/riomh_config.yaml train_net.py --cfg configs/lsun_horse.yaml