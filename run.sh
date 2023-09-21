agent="argue_review"
 
# for dataset in "AQuQ" "GSM8K" "Date" "SocialQA" "CauseEffect" "Objects" "OddOneOut" "LastLetter"; do
for dataset in "Objects"; do
    # for model in "Vicuna-13b" "Vicuna-33b" "Vicuna-7b" "gpt-3.5-turbo" "gpt-4-0613"; do
    for model in "Vicuna-13b"; do
        python main.py --dataset_name $dataset \
              --input data/$dataset/test.jsonl \
              --output data/$dataset/output/test/$agent/$model.jsonl \
              --model_name $model \
              --agent_mode $agent \
              --prompt_version 0 \
              --temperature 0.01 \
              --env cot \\
              --use_wandb
    done
done