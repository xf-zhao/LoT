import json
import re
import csv
from collections import Counter
from fractions import Fraction
import math


# Acknowledgement to https://github.com/veronica320/Faithful-COT
def load_data(frn):
	'''Load data from a file.
	:param frn (str): The dataset file name.

	:return: The dataset (a list of examples, each as a dictionary).
	'''
	if frn.endswith(".jsonl"):
		with open(frn, 'r') as fr:
			lines = []
			for i, line in enumerate(fr):
				if line.strip() == "":
					continue
				try:
					lines.append(json.loads(line))
				except json.decoder.JSONDecodeError as e:
					print(f"Error in line {i}: {line}\n {e}")
					exit(-1)
		return lines
	elif frn.endswith(".csv"):
		with open(frn) as fr:
			reader = csv.DictReader(fr)
			return [line for line in reader]


class GSM8KDataset:
    def __init__(self, data_path) -> None:
        self.data = load_data(data_path)

    def __iter__(self):
        for idata in self.data:
            yield self._parse_data(idata)

    def _parse_data(self, idata):
        question, answer = idata['question'], idata['answer']
        answer_reasoning, answer_num = answer.split('#### ')
        return question, (answer_reasoning, answer_num)
