from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from logithoughts.utils import GSM8KDataset, Metrics
from logithoughts.models import LogiAgent, ThoughtEnv

gsm8k_data_path = "data/GSM8K/"
dataset_test = GSM8KDataset(data_path=gsm8k_data_path + "test.jsonl")
dataset_dev = GSM8KDataset(data_path=gsm8k_data_path + "dev.jsonl")  # for debug
metrics = Metrics(dataset_name="GSM8K_dev")

# todo: Add try-catch another temperature when exception happens
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1024,
)
agent = LogiAgent(chat=chat, check_refined=False)
env = ThoughtEnv(
    chat=chat,
    max_steps=20,
    output=gsm8k_data_path + "output_test.jsonl",
    # output=gsm8k_data_path + "output.jsonl",
    debug=False,
)


for idx, data in tqdm(dataset_test):
    state, terminate = env.reset(data)
    # state = state[0], "1+1=3"
    for i in range(100):
        if terminate:
            metrics.update(env.data)
            metrics.log()
            break
        action = agent.policy(state)
        state, terminate = env.step(action)
    if idx > 100:
        break
pass
