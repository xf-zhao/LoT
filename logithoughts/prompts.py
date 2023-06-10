sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

PT_prompt = """P="{P}".
The reason P is true is that"""

PF_prompt = """P="{P}".
The reason P is false is that"""

PTF_prompt = """P="{P}".
A: P is true because {PT}
B: P is false because {PF}


Which statement (A or B) is correct according to the context?
Therefore, P is true or false?"""