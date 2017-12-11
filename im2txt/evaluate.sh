MSCOCO_DIR="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/data/mscoco"
INCEPTION_CHECKPOINT="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/model/train"
MODEL_DIR="${HOME}/git-reposits/ShowAndTell/ShowAndTell-practice/im2txt/model"
echo ${INCEPTION_CHECKPOINT}
echo ${MODEL_DIR}
python3 evaluate.py \
    --input_file_pattern="${MSCOCO_DIR}/val-?????-of-00004" \
    --eval_dir="${MODEL_DIR}/eval" \
    --checkpoint_dir="${INCEPTION_CHECKPOINT}"
