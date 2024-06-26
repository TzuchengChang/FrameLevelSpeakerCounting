gpus=0
export CUDA_VISIBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
CONFS=conf/train.yml

mkdir -p $EXP_DIR
cp -r local $EXP_DIR/code
python local/train.py --conf_file $CONFS --log_dir $EXP_DIR --gpus $gpus