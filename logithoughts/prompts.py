from easydict import EasyDict

sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

ARGUE_PT = """
Clarification of the next step:
#{col}. {P}

(Note the examined step doesn't have to tackle the whole problem at once.)
Let's clarify the reasoning step #{col}: step #{col} is true because """

# PF_prompt = PT_prompt.replace('is true because', 'is false because')

ARGUE_PF = """
Criticisim of the next step:
#{col}. {P}

(Note the examined step doesn't have to tackle the whole problem at once.)
Let's criticize the reasoning step #{col} for the the conflictions with the quesiont context or with previous steps one by one.
Step #{col} is false because """


ARGUE_PJ = """
Verification of the next step:
#{col}. {P}

Let's check two different reviews (A and B) and pick the correct one to adopt.
Keep in mind the principles I. and II. when analyzing the plausibility of the reviews.
(Note the examined step doesn't have to tackle the whole problem at once.)
I. What are the premises to support the verification of step #{col}? (Your answer should involve exact quote as support.)
II. Does the review actually conflict with the question description or context? Check one by one.
Check carefully and select the correct review. Then, identify whether step #{col} is true or false.

Review A: <review> step #{col} is true because {PT} </review>

Review B: <review> step #{col} is false because {PF} </review>

Analysis and conclusion:
"""
# Candidates:
# Which review is correct? Therefore, step #{col} is true or false?
# Which review is more plausible? Based on this: say whether step #{col} is true or false.

# {col}. It is false to say "{P}" because "{PF}"

ARGUE_PR = """
Revision for the next step:
Original next step #{col}: {P}

(Hint: It is incorrect to directly adopt the step #{col}.)
Let's revise based on the known premises from the question context and from previous verified reasoning steps.
Revision of step #{col}:
"""
 

# It is not correct to directly adopt the step #{col} because {PF}.

Answer_prompt = "Therefore, the numerical (int or float) result is: "

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
        "PR": ARGUE_PR,
    }
)


# reuse
NEGATION_PF = """
#{col}. {P}

Step #{col} is false because """


NEGATION_PJR = """
#{col}. {P}

For the solution step #{col}, let's examine a critical review:
<review> step #{col} is false because {PF} </review>

Analyze carefully the review according to the original question context.
(Your answer should explicitly quote evidence from the question context and previous verified reasoning steps as supports.)
(Your answer should follow the following format.)
Because ____, so the review is correct. Revision of step #{col}: ____.
or Because ____, so the review is incorrect. There is no need for revision.
"""
# Candidates:
# Based on the plausibility of the reviews, say whether step #{col} is true or false?
# Which review is correct? Therefore, step #{col} is true or false?

# {col}. It is false to say "{P}" because "{PF}"

NEGATION_PR = """
#{col}. {P}

Let's revise the step #{col} for a reliable solution.
According to the criteria a~d,
a. The relevant premises (quote exact text in question or step number) related to the statement in this step;
b. An incorrectness in stating the problem or lack of contribution in making progress towards the solution;
c. A conflict with the question setting or previous reasoning steps;
d. A mathematical calculation error.
It is not good to directly adopt the step #{col} because there is an opinion saying: <opinion> It is not true because{PF} </opinion>.
Therefore, step #{col} should be revised as:\n"""


NEGATION = EasyDict(
    {
        "PF": NEGATION_PF,
        "PJR": NEGATION_PJR,
    }
)
