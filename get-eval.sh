export HF_ENDPOINT="https://hf-mirror.com"

python eval.py --input_path xxx/cjo_test/part1.jsonl --output_path xxx/output_115/cjo_train100_part1_TT.jsonl --visible_gpus 0 --use_vllm --resume --use_react --eval_prompt 0 --model_id xxx/Qwen3-8B/xxx --gpu_memory_utilization 0.4 --retrieve_statute True --retrieve_guideline True &
python eval.py --input_path xxx/cjo_test/part2.jsonl --output_path xxx/output_115/cjo_train100_part2_TT.jsonl --visible_gpus 1 --use_vllm --resume --use_react --eval_prompt 0 --model_id xxx/Qwen3-8B/xxx --gpu_memory_utilization 0.4 --retrieve_statute True --retrieve_guideline True &
python eval.py --input_path xxx/cjo_test/part3.jsonl --output_path xxx/output_115/cjo_train100_part3_TT.jsonl --visible_gpus 2 --use_vllm --resume --use_react --eval_prompt 0 --model_id xxx/Qwen3-8B/xxx --gpu_memory_utilization 0.4 --retrieve_statute True --retrieve_guideline True &
python eval.py --input_path xxx/cjo_test/part4.jsonl --output_path xxx/output_115/cjo_train100_part4_TT.jsonl --visible_gpus 3 --use_vllm --resume --use_react --eval_prompt 0 --model_id xxx/Qwen3-8B/xxx --gpu_memory_utilization 0.4 --retrieve_statute True --retrieve_guideline True &

wait