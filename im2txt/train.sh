MSCOCO_DIR="/tmp/mscoco"
HOME="/home/minhopark2115/git-reposits/ShowAndTell/ShowAndTell-practice"
INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt"
MODEL_DIR="${HOME}/im2txt/model"

python3 train.py\
    --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256"\
    --inception-checkpoint-file="${INCEPTION_CHECKPOINT}"\
    --train_dir="${MODEL_DIR}/train" \
    --train_inception=false \
    --number_of_steps=1000000

