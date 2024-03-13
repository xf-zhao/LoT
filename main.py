import argparse
import openai
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from logithoughts.utils import DATASETS, Metrics, BaseMetrics
from logithoughts.models import LogiAgent, CoTEnv, LogiCoTEnv


parser = argparse.ArgumentParser(
    prog="LogiThoughts",
    description="Run thoughts on dataset.",
    epilog="Text at the bottom of help",
)
parser.add_argument("--env", default="cot")
parser.add_argument("--seed", default="1")
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
    # todo: Add try-catch another temperature when exception happens
    chat = ChatOpenAI(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.env == "cot":
        env = CoTEnv(
            chat=chat,
            dataset_name=args.dataset_name,
            output=args.output,
            prompt_version=args.prompt_version,
            debug=args.debug,
        )
        agent = None
        metrics = BaseMetrics(**vars(args))
    elif args.env == "lot":
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
        metrics = Metrics(**vars(args))
    else:
        raise NotImplementedError

    for idx, data in tqdm(dataset):
        del idx
        state, terminate = env.reset(data)
        for _ in range(100):
            if terminate:
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
        openai.api_base = "http://localhost:8000/v1"
    elif args.model_name == "Vicuna-13b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
    elif args.model_name == "Vicuna-7b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
    main(args)
