from langchain.chat_models import ChatOpenAI
from logithoughts.utils import GSM8KDataset
from logithoughts.models import LogiAgent, ThoughtEnv


gsm8k_data_path = "data/GSM8K/"
# dataset_train = GSM8KDataset(data_path=gsm8k_data_path + 'train.jsonl')
# dataset_test = GSM8KDataset(data_path=gsm8k_data_path + 'test.jsonl')
dataset_dev = GSM8KDataset(data_path=gsm8k_data_path + "dev.jsonl")  # for debug

# todo: Add try-catch another temperature when exception happens
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=512)

agent = LogiAgent(chat=chat)
env = ThoughtEnv(chat=chat, max_steps=20, output="data.jsonl", render=True)

for data in dataset_dev:
    question, (answer_reasoning, answer_num) = data
    state, terminate = env.reset(question)
    env.render()
    env.log()
    # state = state[0], "1+1=3"
    for i in range(20):
        action = agent.policy(state)
        state, terminate = env.step(action)
        if terminate:
            break
        pass
pass
