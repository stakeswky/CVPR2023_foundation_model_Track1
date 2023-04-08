export CUDA_VISIBLE_DEVICES=0
export FASTREID_DATASETS=data/datasets

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/vitbase_jointtraining_config.py

python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining --gpus="0"  tools/ufo_train.py --config-file ${config} #--resume 


