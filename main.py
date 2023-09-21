import argparse
import openai
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from logithoughts.utils import DATASETS, Metrics
from logithoughts.models import LogiAgent, CoTEnv, LogiCoTEnv


parser = argparse.ArgumentParser(
    prog="LogiThoughts",
    description="Run thoughts on dataset.",
    epilog="Text at the bottom of help",
)
parser.add_argument("--env", default="cot")
parser.add_argument("--input", default="data/GSM8K/dev.jsonl")
parser.add_argument("--output", default="data/GSM8K/outputs/output_dev.jsonl")
parser.add_argument("--dataset_name", default="GSM8K")
parser.add_argument("--max_steps", default=30, type=int)
parser.add_argument("--temperature", default=0, type=float)
parser.add_argument("--agent_mode", default="argue_review")
parser.add_argument("--prompt_version", default=0, type=int)
parser.add_argument("--max_tokens", default=1024, type=int)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--model_name", default="Vicuna-33b")


def main(args):
    Dataset = DATASETS[args.dataset_name]
    dataset = Dataset(data_path=args.input)
    metrics = Metrics(**vars(args))
    # todo: Add try-catch another temperature when exception happens
    chat = ChatOpenAI(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.env == "cot":
        env = CoTEnv()
        agent = None
    elif args.env == "logicot":
        env = LogiCoTEnv(
            chat=chat,
            dataset_name=args.dataset_name,
            max_steps=args.max_steps,
            output=args.output,
            prompt_version=args.prompt_version,
            debug=args.debug,
        )
        agent = LogiAgent(
            chat=chat,
            check_refined=False,
            mode=args.agent_mode,
        )
    else:
        raise NotImplementedError

    for idx, data in tqdm(dataset):
        del idx
        state, terminate = env.reset(data)
        for _ in range(100):
            if terminate:
                env.terminate()
                metrics.update(env.data)
                metrics.log()
                break
            try:
                action = agent.policy(state)
                state, terminate = env.step(action)
            except Exception as e:
                env.recover(e)
                terminate = True
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model_name == "Vicuna-33b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://134.100.39.10:31046/v1"
    elif args.model_name == "Vicuna-13b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://134.100.9.212:8000/v1"
    elif args.model_name == "Vicuna-7b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://134.100.39.10:30876/v1"
    main(args)
