import json



def preprocess(filename):
    with open(filename) as f:
        datas = json.load(f)

    with open(f'{filename}l', 'a+') as f:
        for d in datas:
            d_json = json.dumps(d)
            f.write(d_json + '\n')
    print('done')

def preprocess_bigbench(filename):
    with open(filename) as f:
        datas = json.load(f)

    description = datas["description"]
    if 'task_prefix' in datas.keys():
        prefix = datas['task_prefix']
    else:
        prefix = description
    examples = datas["examples"]
    with open(f'{filename}l', 'a+') as f:
        for j, ex in enumerate(examples):
            ex_input, target_scores = ex['input'], ex['target_scores']
            question = f'{prefix}\n{ex_input}\n'
            chars = 'ABCDEFG'
            keys = target_scores.keys()
            if j % 2 ==0:
                keys = reversed(keys)
            shuffled_target_scores = {key: target_scores[key] for key in keys}
            for i, (k, v) in enumerate(shuffled_target_scores.items()):
                # opt = 'Opt' + chr(ord('A') + i)
                opt = 'Opt' + chars[i]
                if v==1.0:
                    answer = opt
                question += f'{opt}: {k}\n'
            new_example = {'question': question, 'answer':answer}
            ex_json = json.dumps(new_example)
            f.write(ex_json + '\n')
    print('done')

# preprocess('train.json')
# preprocess('test.json')
preprocess_bigbench('./data/CauseEffect/task.json')
# preprocess_bigbench('./data/SocialQA/task.json')
# preprocess_bigbench('./data/OddOneOut/task.json')
# preprocess_bigbench('./data/Objects/task.json')