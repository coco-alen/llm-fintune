export CUDA_VISIBLE_DEVICES=7

    # --dataset_name="tatsu-lab/alpaca" \
    # --dataset_name='wikitext' \
    # --dataset_config_name='wikitext-2-raw-v1' \
    # --dataset_config_name='wikitext-103-v1' \
    # --load_weight "../weight/opt-125m-block-3bit.pth" \
    # --do_train \
    # --do_eval \

python train.py \
    --dataset_name="wikitext" \
    --dataset_config_name='wikitext-103-v1' \
    --model_name_or_path "facebook/opt-125m" \
    --num_train_epochs=3 \
    --block_size=2048 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --optim paged_adamw_32bit \
    --learning_rate=2.4e-3 \
    --torch_dtype bfloat16 \
    --bf16 \
    --output_dir='./output/opt125m-shiftadd3bit-wikitext103v1' \
    --load_weight "../weight/opt-125m-block-3bit.pth" \
    --do_train \
    --do_eval \
    --evaluation_strategy='epoch' \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_first_step \
    --logging_steps=20 \
    --save_total_limit=1 \
    --run_name='opt-alpaca' \
    --overwrite_output_dir \
    --low_cpu_mem_usage \
    --bits 16 \
    --lora_r 16 \
    --lora_dropout 0.01