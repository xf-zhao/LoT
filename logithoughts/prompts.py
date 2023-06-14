sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

PT_prompt = """
(Let's examine the following step based on the question and current thinking progress. Note that all the required information can be definitely found in the context, and the examined step doese not necessarily tackle the question at once. Evaluate whether it correctly states the problem or contributes to making progress.)
Consider the next step #{col}="{P}".
Step #{col} is true because """

PF_prompt = """
(Let's examine the following step based on the question and current thinking progress. Note that all the required information can be definitely found in the context, and the examined step doese not necessarily tackle the question at once. Evaluate whether it correctly states the problem or contributes to making progress.)
Consider the next step #{col}="{P}".
Step #{col} is false because """

PTF_prompt = """Chose one. According to the question and thinking progress so far, which of the following is correct?
For the next step #{col}="{P}",
A: step #{col} is true because {PT}
B: step #{col} is false because {PF}


Which statement (A or B) is correct?
Therefore, step #{col} is true or false?
(Your answer should follow the following format)
Because ____, so statement ____ is correct. Therefore, step #{col} is ____.
"""

PFR_prompt = """For the next step #{col}, after double-check, it is false to think that "{P}" (because {PF}).
According to the context, step #{col} should be (revised as):\n"""

Answer_prompt = "Therefore, the numerical (int or float) result is: "
