from langchain.chat_models import ChatOpenAI
from logithoughts.utils import GSM8KDataset
from logithoughts.models import LogiAgent, ThoughtEnv


gsm8k_data_path = 'data/GSM8K/'
# dataset_train = GSM8KDataset(data_path=gsm8k_data_path + 'train.jsonl')
# dataset_test = GSM8KDataset(data_path=gsm8k_data_path + 'test.jsonl')
dataset_dev = GSM8KDataset(data_path=gsm8k_data_path + 'dev.jsonl')  # for debug


chat = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.7, max_tokens=512)
agent = LogiAgent(chat=chat)
env = ThoughtEnv(chat=chat)

for data in dataset_dev:
    question, (answer_reasoning, answer_num) = data
    state, terminate = env.reset(question)
    for i in range(20):
        action = agent.policy(state)
        state, terminate = env.step(action)
        pass
    break
pass