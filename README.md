This is the repertory for the **LoT** paper:
 **"Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic"**, *[Xufeng Zhao](https://xf-zhao.github.io/), [Mengdi Li](https://mengdi-li.github.io/), [Wenhao Lu](https://lukaswill.github.io/), [Cornelius Weber](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/weber.html), [Jae Hee Lee](https://jaeheelee.gitlab.io/), [Kun Chu](https://kchu.github.io/), [Stefan Wermter](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/wermter.html).* [[arXiv](https://arxiv.org/abs/2309.13339)], to appear at `COLING 2024`.

![](https://github.com/xf-zhao/xf-zhao.github.io/blob/master/assets/img/publication_preview/LoT%20prompting.gif?raw=true)

Abstract:

Recent advancements in large language models have showcased their remarkable generalizability across various domains. However, their reasoning abilities still have significant room for improvement, especially when confronted with scenarios requiring multi-step reasoning. Although large language models possess extensive knowledge, their reasoning often fails to effectively utilize this knowledge to establish a coherent thinking paradigm. These models sometimes show hallucinations as their reasoning procedures are unconstrained by logical principles. Aiming at improving the zero-shot chain-of-thought reasoning ability of large language models, we propose LoT (Logical Thoughts), a self-improvement prompting framework that leverages principles rooted in symbolic logic, particularly Reductio ad Absurdum, to systematically verify and rectify the reasoning processes step by step. Experimental evaluations conducted on language tasks in diverse domains, including arithmetic, commonsense, symbolic, causal inference, and social problems, demonstrate the efficacy of enhanced reasoning by logic.

# Install

```bash
git clone https://github.com/xf-zhao/LoT.git
cd LoT
pip install -r requirements.txt
```

# Quick Test Run

```bash
python main.py --env cot --model_name gpt-3.5-turbo
python main.py --env lot --model_name gpt-3.5-turbo
```

# Run Exepriments
```bash
sh run_cot.sh
sh run_lot.sh
```

# Dataset

GSM8K: <https://github.com/openai/grade-school-math>


# Cite with BibTex

```text
@inproceedings{Zhao23EnhancingZeroShot,
  title = {Enhancing {{Zero-Shot Chain-of-Thought Reasoning}} in {{Large Language Models}} through {{Logic}}},
  author = {Zhao, Xufeng and Li, Mengdi and Lu, Wenhao and Weber, Cornelius and Lee, Jae Hee and Chu, Kun and Wermter, Stefan},
  booktitle = {2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), Turin, Italy},
  year = {2024},
  month = may,
  keywords = {Large Language Models, Chain-of-Thought, Reasoning, Logic, Reductio ad Absurdum},
}
```
