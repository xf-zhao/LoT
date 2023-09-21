import json
import os
import numpy as np
from loguru import logger
import wandb
import math
import re
import csv
from collections import Counter, defaultdict
from fractions import Fraction
from loguru import logger
import numpy as np
from pathlib import Path


# Acknowledgement to https://github.com/veronica320/Faithful-COT/blob/2f204a48073eebed5d939bc59a7766306fb8298c/source/dataset/utils.py
INVALID_ANS = "[invalid]"


def load_data(frn, format="list", mode="r"):
    """Load data from a file.
    :param frn (str): The dataset file name.

    :return: The dataset (a list of examples, each as a dictionary).
    """
    if not os.path.exists(frn):
        filename = Path(frn)
        filename.parent.mkdir(parents=True, exist_ok=True)
        return []
    if frn.endswith(".jsonl") or frn.endswith(".txt"):
        with open(frn, mode) as fr:
            rtns = [] if format == "list" else {}
            for i, line in enumerate(fr):
                if line.strip() == "":
                    continue
                try:
                    line = json.loads(line)
                    if format == "list":
                        rtns.append(line)
                    else:
                        idx = line["idx"]
                        rtns[idx] = line
                except json.decoder.JSONDecodeError as e:
                    print(f"Error in line {i}: {line}\n {e}")
                    exit(-1)
        return rtns
    elif frn.endswith(".csv"):
        with open(frn) as fr:
            reader = csv.DictReader(fr)
            return [line for line in reader]
    elif frn.endswith(".json"):
        with open(frn, mode) as fr:
            rtns = json.load(fr)
        return rtns


def str2num(answer_str, rounding="int", abs_val=True):
    """Convert a string to a number.
    @:param answer_str (str): The string to convert.
    @:param rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
    @:param abs_val (bool): Whether to take the absolute value of the answer.

    @:return The converted number.
    """
    if "/" in answer_str:
        answer_str = float(sum(Fraction(s) for s in answer_str.split()))
    answer_str = float(answer_str)

    if rounding == "int":
        answer_str = int(answer_str)
    elif rounding == "ceil":
        answer_str = math.ceil(answer_str)
    elif rounding == "floor":
        answer_str = math.floor(answer_str)

    if abs_val:
        answer_str = abs(answer_str)

    return answer_str


def extract_gold_answer(dataset_name, gold_completion):
    """Extract the gold answer from a completion.
    :param dataset_name (str): The name of the dataset.
    :param gold_completion (str): The gold completion.

    :return: The gold answer.
    """
    if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(gold_completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return int(match_str)
        else:
            return INVALID_ANS
    elif dataset_name == "ASDiv":
        # ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
        if type(gold_completion) in [
            tuple,
            list,
        ]:  # the gold answer has multiple values
            answer = dict(Counter([int(ans) for ans in gold_completion]))
        else:  # the gold answer has a single value
            answer = int(gold_completion)
        return answer
    elif dataset_name in ["Date", "CLUTRR"]:
        answer = gold_completion.split("#### ")[-1]
        return answer
    elif dataset_name == "saycan":
        answer = eval(gold_completion)
        return answer
    elif dataset_name in ["StrategyQA"]:
        answer = bool(gold_completion)
        return answer
    elif dataset_name in ["sports"]:
        answer = bool(int(gold_completion))
        return answer
    elif dataset_name in ["LogiQA"]:
        answer = int(gold_completion)
        return answer
    elif dataset_name in ["AQuA", "CauseEffect", "SocialQA", "OddOneOut", "Objects"]:
        gold_completion = gold_completion.replace("Opt", "")
        answer = ord(gold_completion.lower()) - ord("a")
        return answer
    elif dataset_name in ["LastLetter"]:
        return gold_completion
    else:
        return gold_completion


def extract_pred_answer(dataset_name, pred_completion, rounding="int", abs_val=True):
    """Extract the predicted answer from a completion.
    :param dataset_name (str): The name of the dataset.
    :param pred_completion (str): The predicted completion.
    :param rounding (str): The rounding method for the predicted answer. Can be "int", "ceil", or "floor".
    :param abs_val (bool): Whether to take the absolute value of the predicted answer.

    :return: The predicted answer.
    """
    if INVALID_ANS in str(pred_completion):
        return INVALID_ANS

    if dataset_name in ["GSM8K", "SVAMP", "MultiArith"]:
        # GSM8K, SVAMP, and MultiArith all have a single-value integer answer
        if type(pred_completion) == int:
            pred_answer = pred_completion
        elif type(pred_completion) == str:
            ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
            match = ANS_RE.search(pred_completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                try:
                    pred_answer = str2num(match_str, rounding, abs_val)
                except:
                    pred_answer = INVALID_ANS
            else:
                pred_answer = INVALID_ANS
        return pred_answer

    elif dataset_name in ["ASDiv"]:
        # ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
        if type(pred_completion) == int:
            return pred_completion
        elif type(pred_completion) == str:
            pred_completion = pred_completion.lstrip("{([").rstrip("]})")
            pred_answers = pred_completion.split(",")
            final_pred_answers = []
            for pred_answer in pred_answers:
                pred_answer = pred_answer.strip().split(":")[-1].strip("'\"")
                try:
                    pred_answer = str2num(pred_answer, rounding, abs_val)
                    final_pred_answers.append(pred_answer)
                except ValueError:
                    continue
            if len(final_pred_answers) > 1:
                return dict(Counter(final_pred_answers))
            elif len(final_pred_answers) == 1:
                return final_pred_answers[0]
            else:
                return INVALID_ANS
        elif type(pred_completion) == dict:
            new_dict = {}
            for key, value in pred_completion.items():
                new_key = str(key)
                new_key = str2num(new_key, rounding, abs_val)
                new_dict[new_key] = value
            return new_dict

    elif dataset_name in ["StrategyQA"]:
        answer = bool(pred_completion)
        return answer

    elif dataset_name in ["sports"]:
        answer = bool(int(pred_completion))
        return answer

    elif dataset_name in ["saycan"]:
        answer = pred_completion.strip()
        return answer

    # added by xf
    elif dataset_name in ["LogiQA"]:
        ANS_RE = re.compile(r"Opt([a-zA-D])")
        finds = ANS_RE.findall(pred_completion)
        try:
            find = finds[0].lower()
            answer = ord(find) - ord("a")
        except:
            answer = INVALID_ANS
        return answer
    elif dataset_name in ["AQuA", "CauseEffect", "SocialQA", "OddOneOut", "Objects"]:
        ANS_RE = re.compile(r"Opt([a-fA-F])")
        finds = ANS_RE.findall(pred_completion)
        try:
            find = finds[0].lower()
            answer = ord(find) - ord("a")
        except:
            answer = INVALID_ANS
        return answer
    elif dataset_name in ["Date"]:
        ANS_RE = re.compile(r".*(\d\d/\d\d/\d\d\d\d).*")
        finds = ANS_RE.findall(pred_completion)
        try:
            answer = finds[0].lower()
        except:
            answer = INVALID_ANS
        return answer
    else:
        return pred_completion


# Acknowledgement to https://github.com/lz1oceani/verify_cot/blob/main/utils/misc.py#L97
def compare_results(answers, final_answer):
    def compare(answer, final_answer):
        if isinstance(answer, str) and isinstance(final_answer, str):
            correctness = f"{answer}".strip() == f"{final_answer}".strip()
        else:
            try:
                correctness = np.abs(answer - final_answer) < 1e-6
            except Exception as e:
                logger.error(f"Exception: {e}\n Answer is {answer}")
                correctness = f"{answer}".strip() == f"{final_answer}".strip()
        return correctness

    if not isinstance(answers, list):
        return compare_results([answers], final_answer)[0]
    ret = [compare(answer, final_answer) for answer in answers]
    return ret


class BaseMetrics:
    def __init__(self, **kwargs) -> None:
        self._idxs = []
        self._data = None
        self._correctnesses = []
        self.use_wandb = kwargs["use_wandb"]
        if self.use_wandb:
            wandb.init(
                project="LogiCoT",
                # Track hyperparameters and run metadata
                config=kwargs,
            )

    @property
    def correctnesses(self):
        return np.array(self._correctnesses)

    @property
    def acc(self):
        return self.correctnesses.mean(axis=0)

    @property
    def reports(self):
        reports = {
            "idx": self._data["idx"],
            "acc": self.acc[0],
        }
        return reports

    def update(self, data):
        idx = data["idx"]
        self._idxs.append(idx)
        answer_default, answer = (
            data["answer_default"],
            data["answer"],
        )
        correctness = compare_results(answers=answer_default, final_answer=answer)
        self._correctnesses.append(correctness)
        self._data = data
        return

    def log(self):
        if self.use_wandb:
            reports = self.reports
            wandb.log(reports)
        logger.debug(self.reports)
        return


class Metrics(BaseMetrics):
    def __init__(self, **kwargs) -> None:
        self.rec = re.compile(r"T\[(\d+), (\d+)\]")
        self._final_steps = []
        self._final_layer = []
        self._default_steps = []
        super().__init__(**kwargs)

    @property
    def improve_rate(self):
        c = self.correctnesses
        # #[0,1]/#[0,*]
        cF = c[c[:, 0] == False]
        if len(cF) > 0:
            return cF[:, 1].mean()
        return -1

    @property
    def worse_rate(self):
        c = self.correctnesses
        # #[1,0]/#[1,*]
        cF = c[c[:, 0] == True]
        if len(cF) > 0:
            return (~cF[:, 1]).mean()
        return -1

    @property
    def reports(self):
        reports = {
            "idx": self._data["idx"],
            "acc_default": self.acc[0],
            "acc_logi": self.acc[1],
            "improve rate": self.improve_rate,
            "worse rate": self.worse_rate,
            "T_default": int(self._correctnesses[-1][0]),
            "T_logi": int(self._correctnesses[-1][1]),
            "steps_default": np.array(self._default_steps).mean(),
            "steps_logi": np.array(self._final_steps).mean(),
            "layers_logi": np.array(self._final_layer).mean(),
        }
        return reports

    def update(self, data):
        idx = data["idx"]
        self._idxs.append(idx)
        answer_default, answer_logi, answer = (
            data["answer_default"],
            data["answer_logi"],
            data["answer"],
        )
        correctness = compare_results(
            answers=[answer_default, answer_logi], final_answer=answer
        )
        self._correctnesses.append(correctness)

        # counting steps
        nodes = data["G"]["nodes"]
        steps = defaultdict(list)
        for node in nodes:
            nid = node["id"]
            match = self.rec.search(nid)
            m, n = int(match[1]), int(match[2])
            steps[m].append(n)
        default_layer, final_layer = min(steps.keys()), max(steps.keys())
        default_steps, final_steps = max(steps[default_layer]), max(steps[final_layer])
        self._final_layer.append(final_layer)
        self._default_steps.append(default_steps)
        self._final_steps.append(final_steps)
        self._data = data
        return


class Dataset:
    def __init__(self, data_path) -> None:
        self.data = load_data(data_path)

    def __iter__(self):
        raise NotImplementedError

    def _parse_data(self, *args, **kwargs):
        raise NotImplementedError


class LogiQADataset(Dataset):
    def __iter__(self):
        for _, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                idx = int(idata["id"])
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        only_question, text, answer = idata["question"], idata["text"], idata["answer"]
        options = idata["options"]
        options = [f"Opt{i}: {opt}" for i, opt in zip("ABCD", options)]
        option_text = "\n".join(options)
        question = text + only_question + "\nOptions:\n" + option_text
        gold_answer = extract_gold_answer("LogiQA", answer)
        data = {
            "question": question,
            "answer": gold_answer,
        }
        return data


class GSM8KDataset(Dataset):
    def __iter__(self):
        for idx, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("GSM8K", answer)
        return {"question": question, "answer": gold_answer}


class AQuADataset(Dataset):
    def __iter__(self):
        for idx, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["correct"]
        gold_answer = extract_gold_answer("AQuA", answer)
        options = idata["options"]
        options = [f"Opt{opt}" for opt in options]
        option_text = "\n".join(options)
        question = question + "\nOptions:\n" + option_text
        data = {
            "question": question,
            "answer": gold_answer,
        }
        return data


class DateDataset(GSM8KDataset):
    # Dataset from: https://github.com/veronica320/Faithful-COT/blob/main/data/date/test.jsonl
    def __iter__(self):
        for idata in self.data:
            try:
                data = self._parse_data(idata)
                idx = idata["id"]
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("Date", answer)
        return {"question": question, "answer": gold_answer}


class LastLetterDataset(GSM8KDataset):
    # Dataset from https://huggingface.co/datasets/ChilleD/LastLetterConcat

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("LastLetter", answer)
        return {"question": question, "answer": gold_answer}


class CauseEffectDataset(GSM8KDataset):
    # https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/cause_and_effect/one_sentence/task.json

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("CauseEffect", answer)
        return {"question": question, "answer": gold_answer}


class SocialQADataset(CauseEffectDataset):
    # https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/social_iqa/task.json

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("SocialQA", answer)
        return {"question": question, "answer": gold_answer}


class OddOneOutDataset(CauseEffectDataset):
    # https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/odd_one_out/task.json
    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("OddOneOut", answer)
        return {"question": question, "answer": gold_answer}


class ObjectsDataset(CauseEffectDataset):
    # https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/tracking_shuffled_objects/three_objects/task.json
    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        gold_answer = extract_gold_answer("Objects", answer)
        return {"question": question, "answer": gold_answer}


DATASETS = {
    "GSM8K": GSM8KDataset,
    "LogiQA": LogiQADataset,
    "AQuA": AQuADataset,
    "Date": DateDataset,
    "LastLetter": LastLetterDataset,
    "CauseEffect": CauseEffectDataset,
    "SocialQA": SocialQADataset,
    "OddOneOut": OddOneOutDataset,
    "Objects": ObjectsDataset,
}
