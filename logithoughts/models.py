import re
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)
import networkx as nx
from logithoughts.prompts import sys_prompt,PT_prompt, PF_prompt, PTF_prompt



class ThoughtEnv:
    def __init__(self, chat, max_steps=15) -> None:
        self.chat = chat
        self.max_steps = max_steps
        self.i_step = 0
        self.P = None
        self.sys_msg =None
        self.thoughts = None
        self.thoughts_spliter = re.compile(r'\n\d\.\s') 
        self.sys_prompt_template = SystemMessagePromptTemplate.from_template(sys_prompt)
        self.G = None

    def reset(self, question):
        self.i_step = 0
        self.G = nx.Graph()
        # print('*'*50 + ' '*10 + f'RESET' + ' '*10 + '*'*50)
        self.sys_msg = self.sys_prompt_template.format(question=question)
        resp = self.chat([self.sys_msg, SystemMessage(content='1. ')])
        thoughts = self.thoughts_spliter.split(resp.content)
        P = thoughts[self.i_step]
        terminate = len(thoughts) <= 1
        self.thoughts = thoughts
        self.P = P.replace('"', "'")
        state = self.sys_msg, self.P
        return state, terminate

    def step(self, action):
        self.i_step+=1
        print('*'*50 + ' '*10 + f'CHECKED {self.i_step}, STEP INTO {self.i_step + 1}' + ' '*10 + '*'*50)
        advice, run_again = action
        sys_msg = SystemMessage(content=self.sys_msg.content+f"{self.i_step}. {advice}\n")
        self.sys_msg = sys_msg
        if run_again:
            resp = self.chat([self.sys_msg])
            new_thoughts_str =resp.content
            new_thoughts = self.thoughts_spliter.findall(new_thoughts_str)
            thoughts = self.thoughts[:self.i_step] + new_thoughts
            self.thoughts = thoughts

        if self.i_step >= min(self.max_steps, len(self.thoughts)):
            # TODO: add final request for answer checking
            self.P = None
            terminate = True
        else:
            self.P = self.thoughts[self.i_step].replace('"', "'")
            terminate = False
        state = self.sys_msg, self.P
        return state, terminate 


class LogiAgent:
    def __init__(self, chat) -> None:
        self.chat = chat
        self.PT_prompt_template = HumanMessagePromptTemplate.from_template(PT_prompt)
        self.PF_prompt_template = HumanMessagePromptTemplate.from_template(PF_prompt)
        self.PTF_prompt_template = HumanMessagePromptTemplate.from_template(PTF_prompt)

    def policy(self, state):
        sys_msg, P = state
        PT_msg = self.PT_prompt_template.format(P=P)
        msgs = [sys_msg, PT_msg]
        PT  = self.chat(msgs).content
        print('-'*100)
        print(PT)

        PF_msg = self.PF_prompt_template.format(P=P)
        msgs = [sys_msg, PF_msg]
        PF  = self.chat(msgs).content
        print('-'*100)
        print(PF)

        PTF_msg = self.PTF_prompt_template.format(P=P, PT=PT, PF=PF)
        print('-'*100)
        print(PTF_msg.content)
        msgs = [sys_msg, PTF_msg]
        choice  = self.chat(msgs).content
        print('-'*100)
        print(choice)
        if 'p is true' in choice.lower():
            advice = P
            run_again = False
        elif 'p is false' in choice.lower():
            advice = f'It is not true to say that "{P}" because {PF}'
            run_again = True
        else:
            print(choice)
            raise Exception
        return advice, run_again