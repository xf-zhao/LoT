sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

PT_prompt = """
Examine ideas relevent to solve this question. Remember, it's not necessary to tackle the entire question at once. Instead, concentrate on evaluating whether each idea correctly states the problem or contributes to making progress in solving the problem.
P="{P}".
P is true because """

PF_prompt = """
Examine ideas relevent to solve this question. Remember, it's not necessary to tackle the entire question at once. Instead, concentrate on evaluating whether each idea correctly states the problem or contributes to making progress in solving the problem.
P="{P}".
P is false because """

PF_refined_prompt = """
P="{P}".
P is false because {PF}.
Therefore, the refinement is: """

PTF_prompt = """
Let's check two opinions that that are relevant to resolving this question.
P="{P}".
A: P is true because {PT}
B: P is false because {PF}


Which statement (A or B) is correct?
Therefore, P is true or false?
(Your answer should follow the following format)
Because ____, so statement ____ is correct. Therefore, P is ____.
"""