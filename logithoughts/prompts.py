from easydict import EasyDict

base_sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

base_sys_prompt_logi = """Question: {question}

Let's prove by the mean of "reduction to absurdity": Negate conclusion and conjunct with premises to determine whether the conjunction is a contradiction. If so, that is the option to select.
Let's think step by step.
Answer:
"""


SYS_PROMPTS = {
    "GSM8K": {0: base_sys_prompt},
    "LogiQA": {
        0: "Analyze and answer the following single-choice problem in the symbolic logic field.\n"
        + base_sys_prompt,
        1: "Analyze and answer the following single-choice problem in the symbolic logic field.\n"
        + base_sys_prompt_logi,
    },
    # + base_sys_prompt.replace("step by step", "step by step for every Opt"),
    "AQuA": {
        0: "Analyze and answer the following single-choice problem.\n" + base_sys_prompt
    },
    "Date":{0:base_sys_prompt},
    'LastLetter':{0:base_sys_prompt},
    'CauseEffect':{0 :base_sys_prompt},
    'SocialQA':{0 :base_sys_prompt},
    'OddOneOut':{0 :base_sys_prompt},
    'Objects':{0 :base_sys_prompt},
}

ANSWER_PROMPTS = EasyDict(
    {
        "LogiQA": "Therefore, the final answer is (chose only one option indicator from the list [OptA, OptB, OptC, OptD]):",
        "GSM8K": "Therefore, the numerical (int or float) result is: ",
        "AQuA": "Therefore, the final answer is (chose only one opiton indicator from the list [OptA, OptB, OptC or OptD]):",
        'Date':"Therefore, the answer (in MM/DD/YYYY format) is:",
        'LastLetter': "Therefore, the answer (only the answer no extra comments) is:",
        "CauseEffect": "Therefore, the final answer is (chose only one option indicator from the list [OptA, OptB]):",
        "SocialQA": "Therefore, the final answer is (chose only one option indicator from the list [OptA, OptB, OptC]):",
        "OddOneOut": "Therefore, the final answer is (chose only one option indicator from the list [OptA, OptB, OptC, OptD, OptE, OptF]:",
        "Objects": "Therefore, the final answer is (chose only one option indicator from the list [OptA, OptB, OptC]):",
    }
)

ARGUE_PT = """
Clarification of the next step:
#{col}. {P}

Let's clarify the reasoning step #{col} for the the consistency with the quesion premises or with previous reasoning steps.
Step #{col} is true because """

# PF_prompt = PT_prompt.replace('is true because', 'is false because')

ARGUE_PF = """
Criticisim of the next step:
#{col}. {P}

Let's criticize the reasoning step #{col} for the the conflictions to the quesion premises or to previous reasoning steps.
Step #{col} is false because """


ARGUE_PJ = """
Verification of the next step:
#{col}. {P}

Let's check two different reviews (X and Y).
Suport the more plausible one and criticise the other one.
Review X: <review> step #{col} is TRUE because {PT} </review>
Review Y: <review> step #{col} is FALSE because {PF} </review>

Let's start by analyzing one by one:
I. What are the premises and previous steps to support the verification of step #{col}? (Your answer should quote exact quote as support.)
II. Criticise the incorrect review.
(Note the examined step doesn't have to tackle the whole problem at once.)
Finally, identify whether step #{col} is true or false.

Analysis and conclusion:
"""
# Candidates:
# Which review is correct? Therefore, step #{col} is true or false?
# Which review is more plausible? Based on this: say whether step #{col} is true or false.

# {col}. It is false to say "{P}" because "{PF}"

ARGUE_PR_NOREIVEW = """
Revision for the next step:
Original next step #{col}: {P}

(Hint: It is not good to directly adopt the step #{col}.)
Let's revise for a better version based on the question premises and on the reasoning steps so far.
The revision should keep the question and options intact. Only revise the reasoning process.
Revision of step #{col}:
"""

ARGUE_PR = """
Revision for the next step:
Original next step #{col}: {P}

(Hint: It is not good to directly adopt the step #{col} because there is a review says <review> {PF} </review>.)
Let's revise for a better version based on the question premises and on the reasoning steps so far.
Revision of step #{col}:
"""


# It is not correct to directly adopt the step #{col} because {PF}.


# Which statement (A or B) is correct?
# Therefore, step #{col} is true or false?
# Therefore, by reiterating or by revising, the correct step #{col} is: ____.
# PF_prompt = """
# Let's think one step further to solve the problem.
# (Note the examined step doesn't have to tackle the whole problem at once.)
# (You should verify the correctness of mathmatical calculation if there is any.)
# (You will focus whether it correctly states the problem or contributes to makeing a progress.)
# For the next step #{col}=<step> {P} </step>, step #{col} is false because """
# (Your answer should include explicit quotes as support, and follow the format below.)
# Because ____ (exact quotes and deduction), so review ____ is correct. Therefore, step #{col} is ____.

ARGUE = EasyDict(
    {
        "PT": ARGUE_PT,
        "PF": ARGUE_PF,
        "PJ": ARGUE_PJ,
        "PJ": ARGUE_PJ,
        "PR": ARGUE_PR,
        "PR_N": ARGUE_PR_NOREIVEW,
    }
)


NEGATION_PF = """
#{col}. {P}

Let's verify the step #{col} by the mean of "reduction to absurdity": Negate this step and conjunct with relevant premises and previous reasoning steps to determine whether the conjunction is a contradiction. If so, this step can be verified valid.
Finally, answer whether step #{col} is true or false.
"""
# Let's think step by step.


NEGATION_PR = ARGUE_PR

NEGATION = EasyDict(
    {
        "PF": NEGATION_PF,
        "PR": NEGATION_PR,
    }
)
