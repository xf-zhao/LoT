env="lot"
seed="1"
 
# for dataset in "AQuA" "GSM8K" "Date" "SocialQA" "CauseEffect" "Objects" "OddOneOut" "LastLetter"; do
for dataset in "AQuA"; do
    # for model in "Vicuna-13b" "Vicuna-33b" "Vicuna-7b" "gpt-3.5-turbo" "gpt-4-0613"; do
    for model in "Vicuna-13b"; do
        python main.py --dataset_name $dataset \
              --input data/$dataset/test.jsonl \
              --output data/$dataset/output/$model/${env}_${seed}.jsonl \
              --model_name $model \
              --prompt_version 0 \
              --temperature 0.7 \
              --env $env \
              --seed $seed \
              --use_wandb &
    done
done
