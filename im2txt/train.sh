MSCOCO_DIR="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/data/mscoco"
INCEPTION_CHECKPOINT="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/data/inception_v3.ckpt"
MODEL_DIR="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/model"
echo ${INCEPTION_CHECKPOINT}
echo ${MODEL_DIR}
python3 train.py \
    --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
    --inception_checkpoint_file="${INCEPTION_CHECKPOINT}"\
    --train_dir="${MODEL_DIR}/train" \
    --train_inception=false \
    --number_of_steps=1000000

