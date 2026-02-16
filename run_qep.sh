GPU=$1
CONFIG=$2
DATASET=$3
BITS=$4
GROUP_SIZE=$5
NSAMPLES=$6

# CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method awq-qep --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --save_path ./output/llama-2-7b-QEP-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES
# CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method gptq-qep --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --save_path ./output/llama-2-7b-QEP-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES

# qep
CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method gptq-qep --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --qep --save_path ./output/llama-2-7b-GPTQ-QEP-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES
# CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method awq-qep --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --qep --save_path ./output/llama-2-7b-AWQ-QEP-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES