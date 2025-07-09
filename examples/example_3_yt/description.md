This is a tutorial from youtube https://www.youtube.com/watch?v=XWkUC1Y72Yc

github repo: https://github.com/AIAnytime/Multi-GPU-Fine-Training-LLMs


# Strenghts:

- Uses a tiny llama model (decoder)
- Uses a standard config deepspeed file


# How to run? run this on bash console

accelerate launch --config_file "config.yaml"  train.py --seed 100 --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset_name "omi-health/medical-dialogue-to-soap-summary" --chat_template_format "chatml" --add_special_tokens False --append_concat_token False --splits "train,test" --max_seq_len 2048 --num_train_epochs 1 --logging_steps 5 --log_level "info" --logging_strategy "steps" --eval_strategy "epoch" --save_strategy "no" --bf16 True --packing True --learning_rate 1e-4 --lr_scheduler_type "cosine" --weight_decay 1e-4 --warmup_ratio 0.0 --max_grad_norm 1.0 --output_dir "llama-sft-qlora-dsz3" --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 --gradient_checkpointing True --use_reentrant True --dataset_text_field "content" --use_flash_attn False --use_peft_lora True --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "all-linear" --use_4bit_quantization True --use_nested_quant True --bnb_4bit_compute_dtype "bfloat16" --bnb_4bit_quant_storage_dtype "bfloat16"


