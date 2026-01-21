python judge.py --i /private/My_baseline/SVM/cjo_predictions.jsonl --m svm
python judge.py --i /private/My_baseline/SVM/cail_predictions.jsonl --m svm

python judge.py --i /private/My_baseline/CNN/infer_results/cjo_predictions.jsonl --m cnn
python judge.py --i /private/My_baseline/CNN/infer_results/cail_predictions.jsonl --m cnn
python judge.py --i /private/My_baseline/CNN/infer_results/cail_final_predictions.jsonl --m cnn

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/cjo_predictions.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/cail_predictions.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-Lora-infer/cjo_predictions.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-Lora-infer/cail_predictions.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_1_7B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_1_7B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_4B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_4B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_8B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_8B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_14B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_14B.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Bert/1epoch/cjo_test3.jsonl --m bert
python judge.py --i /private/My_baseline/Bert/1epoch/cail_test3.jsonl --m bert

# Qwen3-8B step 0 
python judge.py --i xxx/output_0926/cjo_raw_part1.jsonl xxx/output_0926/cjo_raw_part2.jsonl
python judge.py --i xxx/output_0926/cail_raw_part1.jsonl xxx/output_0926/cail_raw_part2.jsonl

# Qwen3-14B step 0 
python judge.py --i xxx/output_1004/cjo_raw_14B_part1.jsonl \
                    xxx/output_1004/cjo_raw_14B_part2.jsonl \
                    xxx/output_1004/cjo_raw_14B_part3.jsonl \
                    xxx/output_1004/cjo_raw_14B_part4.jsonl

python judge.py --i xxx/output_1004/cail_raw_14B_part11.jsonl \
                    xxx/output_1004/cail_raw_14B_part12.jsonl \
                    xxx/output_1004/cail_raw_14B_part13.jsonl \
                    xxx/output_1004/cail_raw_14B_part14.jsonl \
                    xxx/output_1004/cail_raw_14B_part21.jsonl \
                    xxx/output_1004/cail_raw_14B_part22.jsonl \
                    xxx/output_1004/cail_raw_14B_part23.jsonl \
                    xxx/output_1004/cail_raw_14B_part24.jsonl
# MSR
python judge.py --i xxx/output_1231/cjo_train660_part1.jsonl \
                    xxx/output_1231/cjo_train660_part2.jsonl \
                    xxx/output_1231/cjo_train660_part3.jsonl \
                    xxx/output_1231/cjo_train660_part4.jsonl

python judge.py --i xxx/output_0108/cail_train_660_part11.jsonl \
                    xxx/output_0108/cail_train_660_part12.jsonl \
                    xxx/output_0108/cail_train_660_part13.jsonl \
                    xxx/output_0108/cail_train_660_part14.jsonl \
                    xxx/output_0108/cail_train_660_part21.jsonl \
                    xxx/output_0108/cail_train_660_part22.jsonl \
                    xxx/output_0108/cail_train_660_part23.jsonl \
                    xxx/output_0108/cail_train_660_part24.jsonl

# w/ process reward

python judge.py --i xxx/output_1006/cjo_train40_star_part1.jsonl \
                    xxx/output_1006/cjo_train40_star_part2.jsonl \
                    xxx/output_1006/cjo_train40_star_part3.jsonl \
                    xxx/output_1006/cjo_train40_star_part4.jsonl --m star

python judge.py --i xxx/output_1008/cjo_train80_star_part1.jsonl \
                    xxx/output_1008/cjo_train80_star_part2.jsonl \
                    xxx/output_1008/cjo_train80_star_part3.jsonl \
                    xxx/output_1008/cjo_train80_star_part4.jsonl --m star

python judge.py --i xxx/output_1009/cjo_train100_star_part1.jsonl \
                    xxx/output_1009/cjo_train100_star_part2.jsonl \
                    xxx/output_1009/cjo_train100_star_part3.jsonl \
                    xxx/output_1009/cjo_train100_star_part4.jsonl --m star  

python judge.py --i xxx/output_1009/cjo_train120_star_part1.jsonl \
                    xxx/output_1009/cjo_train120_star_part2.jsonl \
                    xxx/output_1009/cjo_train120_star_part3.jsonl \
                    xxx/output_1009/cjo_train120_star_part4.jsonl --m star

python judge.py --i xxx/output_1010/cjo_train140_star_part1.jsonl \
                    xxx/output_1010/cjo_train140_star_part2.jsonl \
                    xxx/output_1010/cjo_train140_star_part3.jsonl \
                    xxx/output_1010/cjo_train140_star_part4.jsonl --m star

# w/o process reward

python judge.py --i xxx/output_0112/cjo_train40_part1.jsonl \
                    xxx/output_0112/cjo_train40_part2.jsonl \
                    xxx/output_0112/cjo_train40_part3.jsonl \
                    xxx/output_0112/cjo_train40_part4.jsonl

python judge.py --i xxx/output_0112/cjo_train80_part1.jsonl \
                    xxx/output_0112/cjo_train80_part2.jsonl \
                    xxx/output_0112/cjo_train80_part3.jsonl \
                    xxx/output_0112/cjo_train80_part4.jsonl

python judge.py --i xxx/output_0112/cjo_train100_part1.jsonl \
                    xxx/output_0112/cjo_train100_part2.jsonl \
                    xxx/output_0112/cjo_train100_part3.jsonl \
                    xxx/output_0112/cjo_train100_part4.jsonl

python judge.py --i xxx/output_0112/cjo_train120_part1.jsonl \
                    xxx/output_0112/cjo_train120_part2.jsonl \
                    xxx/output_0112/cjo_train120_part3.jsonl \
                    xxx/output_0112/cjo_train120_part4.jsonl

python judge.py --i xxx/output_0112/cjo_train140_part1.jsonl \
                    xxx/output_0112/cjo_train140_part2.jsonl \
                    xxx/output_0112/cjo_train140_part3.jsonl \
                    xxx/output_0112/cjo_train140_part4.jsonl



# w/o multi-source retrieval
python judge.py --i xxx/output_112/cjo_train40_part1_TF.jsonl \
                    xxx/output_112/cjo_train40_part2_TF.jsonl \
                    xxx/output_112/cjo_train40_part3_TF.jsonl \
                    xxx/output_112/cjo_train40_part4_TF.jsonl

python judge.py --i xxx/output_112/cjo_train60_part1_TF.jsonl \
                    xxx/output_112/cjo_train60_part2_TF.jsonl \
                    xxx/output_112/cjo_train60_part3_TF.jsonl \
                    xxx/output_112/cjo_train60_part4_TF.jsonl

python judge.py --i xxx/output_112/cjo_train80_part1_TF.jsonl \
                    xxx/output_112/cjo_train80_part2_TF.jsonl \
                    xxx/output_112/cjo_train80_part3_TF.jsonl \
                    xxx/output_112/cjo_train80_part4_TF.jsonl

python judge.py --i xxx/output_112/cjo_train100_part1_TF.jsonl \
                    xxx/output_112/cjo_train100_part2_TF.jsonl \
                    xxx/output_112/cjo_train100_part3_TF.jsonl \
                    xxx/output_112/cjo_train100_part4_TF.jsonl

python judge.py --i xxx/output_112/cjo_train120_part1_TF.jsonl \
                    xxx/output_112/cjo_train120_part2_TF.jsonl \
                    xxx/output_112/cjo_train120_part3_TF.jsonl \
                    xxx/output_112/cjo_train120_part4_TF.jsonl

python judge.py --i xxx/output_112/cjo_train140_part1_TF.jsonl \
                    xxx/output_112/cjo_train140_part2_TF.jsonl \
                    xxx/output_112/cjo_train140_part3_TF.jsonl \
                    xxx/output_112/cjo_train140_part4_TF.jsonl

# direct infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-1-7b_cjo_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-1-7b_cail_direct.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-4b_cjo_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-4b_cail_direct.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-14b_cjo_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-14b_cail_direct.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_llama3-8b_cjo_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_llama3-8b_cail_direct.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/cjo_predictions.jsonl --m direct_infer # qwen3-8b
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/cail_predictions.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-32b_cjo_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-235b_cjo_direct.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-32b_cail_direct.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-235b_cail_direct.jsonl --m direct_infer



# direct reasoning

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-32b_cjo_think.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-235b_cjo_think.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-32b_cail_think.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-direct-infer/pred_qwen3-235b_cail_think.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_1_7B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_1_7B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_4B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_4B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_8B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_8B.jsonl --m direct_infer

python judge.py --i /private/legal-R1/cail2018/cjo_predictions_14B.jsonl --m direct_infer
python judge.py --i /private/legal-R1/cail2018/cail_predictions_14B.jsonl --m direct_infer

# SFT

python judge.py --i /private/My_baseline/Qwen3-8B-Lora-infer/cjo_predictions.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Qwen3-8B-Lora-infer/cail_predictions.jsonl --m direct_infer

python judge.py --i /private/My_baseline/Llama3-8B-Lora-infer/cjo_predictions.jsonl --m direct_infer
python judge.py --i /private/My_baseline/Llama3-8B-Lora-infer/cail_predictions.jsonl --m direct_infer