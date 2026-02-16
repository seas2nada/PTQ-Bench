GPU=$1
CONFIG=$2
DATASET=$3
BITS=$4
GROUP_SIZE=$5
NSAMPLES=$6

DATASET=wikitext2
CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method c-gptq --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --save_path ./output/llama-2-7b-GPTQ-$DATASET-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES

PREV_DATASET=$DATASET
DATASET=boolq
H_IN=./output/llama-2-7b-GPTQ-${PREV_DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES/h_out.pt
CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method c-gptq --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --h-in $H_IN --save_path ./output/llama-2-7b-GPTQ-${PREV_DATASET}_${DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES

PREV_DATASET=${PREV_DATASET}_${DATASET}
DATASET=piqa
H_IN=./output/llama-2-7b-GPTQ-${PREV_DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES/h_out.pt
CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method c-gptq --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --h-in $H_IN --save_path ./output/llama-2-7b-GPTQ-${PREV_DATASET}_${DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES

PREV_DATASET=${PREV_DATASET}_${DATASET}
DATASET=winogrande
H_IN=./output/llama-2-7b-GPTQ-${PREV_DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES/h_out.pt
CUDA_VISIBLE_DEVICES=$GPU python run_quant.py --method c-gptq --config $CONFIG --dataset $DATASET --bits $BITS --group_size $GROUP_SIZE --h-in $H_IN --save_path ./output/llama-2-7b-GPTQ-${PREV_DATASET}_${DATASET}-${BITS}bit-g$GROUP_SIZE-nsamples$NSAMPLES
