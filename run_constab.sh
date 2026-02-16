GPU=$1
CONFIG=$2
DATASET=$3
BITS=$4
GROUP_SIZE=$5
NSAMPLES=$6

CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method constab --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --nsamples $NSAMPLES --save_path ./output/llama-2-7b-constab-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES