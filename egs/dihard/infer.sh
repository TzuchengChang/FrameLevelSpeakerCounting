gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
WAVS=F:/data/ami/amicorpus
CKPT=checkpoints/epoch=77-step=71214.ckpt
OUT=$EXP_DIR/preds/${CKPT}

python $EXP_DIR/code/infer.py --exp_dir $EXP_DIR --checkpoint_name $CKPT --wav_dir $WAVS --out_dir $OUT --gpus $gpus --regex Array1-01