import json
import os
import numpy as np
from loguru import logger
import wandb
import math
import re
import csv
from collections import Counter
from fractions import Fraction
from loguru import logger
import numpy as np


# Acknowledgement to https://github.com/veronica320/Faithful-COT/blob/2f204a48073eebed5d939bc59a7766306fb8298c/source/dataset/utils.py
INVALID_ANS = "[invalid]"


def load_data(frn, format="list", mode="r"):
    """Load data from a file.
    :param frn (str): The dataset file name.

    :return: The dataset (a list of examples, each as a dictionary).
    """
    if not os.path.exists(frn):
        return []
    if frn.endswith(".jsonl"):
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
    elif dataset_name in ["date", "CLUTRR"]:
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

    else:
        return pred_completion


class GSM8KDataset:
    def __init__(self, data_path) -> None:
        self.data = load_data(data_path)

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
        reasoning, _ = answer.split("#### ")
        gold_answer = extract_gold_answer("GSM8K", answer)
        return {"question": question, "reasoning": reasoning, "answer": gold_answer}


# Acknowledgement to https://github.com/lz1oceani/verify_cot/blob/main/utils/misc.py#L97
def compare_results(answers, final_answer):
    def compare(answer, final_answer):
        try:
            correctness = np.abs(answer - final_answer) < 1e-6
        except Exception as e:
            logger.error(answer)
            correctness = f"{answer}".strip() == f"{final_answer}".strip
        return correctness

    if not isinstance(answers, list):
        return compare_results([answers], final_answer)[0]
    ret = [compare(answer, final_answer) for answer in answers]
    return ret


class Metrics:
    def __init__(self, **kwargs) -> None:
        self._idxs = []
        self._data = None
        self._correctnesses = []
        self.use_wandb = kwargs['use_wandb']
        if self.use_wandb:
            wandb.init(
                project="LogiThoughts",
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
        self._data = data
        return

    def log(self):
        if self.use_wandb:
            wandb.log(self.reports)
            data = self._data
            table = wandb.Table(
                columns=[
                    "idx",
                    "question",
                    "answer",
                    "answer_default",
                    "answer_logi",
                    "G",
                ]
            )
            idx = data["idx"]
            table.add_data(
                idx,
                data["question"],
                data["answer"],
                data["answer_default"],
                data["answer_logi"],
                data["G"],
            )
            wandb.log({f"Thoughts {idx}": table})
        logger.debug(self.reports)
        return
