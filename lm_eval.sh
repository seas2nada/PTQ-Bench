export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

GPU=$1
MODEL=$2

# TASKS="boolq,piqa,winogrande,hellaswag,arc_easy,arc_challenge,gsm8k"
# TASKS="boolq,piqa,medqa_4options,truthfulqa"
TASKS="boolq,piqa,arc_easy,arc_challenge,medqa_4options,truthfulqa,gsm8k"
CUDA_VISIBLE_DEVICES=$GPU lm_eval --model hf \
        --model_args pretrained=$MODEL \
        --tasks $TASKS \
        --batch_size 1 \
        --output ./results/$MODEL \
        --confirm_run_unsafe_code