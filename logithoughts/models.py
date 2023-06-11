import re
import sys
import json
from loguru import logger
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
import networkx as nx
import matplotlib.pyplot as plt
from .prompts import sys_prompt, PT_prompt, PF_prompt, PTF_prompt, PF_refined_prompt

COLORMAP = {
    "maybe": "yellow",
    "yes": "green",
    "no": "red",
    "comment": "lightgreen",
    "never": "gray",
    "done": "purple",
}


class ThoughtEnv:
    def __init__(
        self, chat, max_steps=25, output=None, render=True, log="INFO"
    ) -> None:
        self.chat = chat
        self.max_steps = max_steps
        self.output = output
        self.thoughts_spliter = re.compile(r"(?:\n|^)\d+\.\s")
        self.sys_prompt_template = SystemMessagePromptTemplate.from_template(sys_prompt)
        self.Gs = []
        self._render = render
        if log == "INFO":
            logger.add(sys.stderr, level="INFO")
        elif log == "DEBUG":
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="ERROR")

    @property
    def index(self):
        return self.get_index(self.row, self.col)

    def reset(self, question):
        self._reset()
        self.sys_msg = self.sys_prompt_template.format(question=question)
        resp = self.chat([self.sys_msg, SystemMessage(content=f"{self.col}. ")])
        chain = self.sys_msg.content + f"{self.col}. " + resp.content
        self.chains.append(chain)
        thoughts = self.thoughts_spliter.split(resp.content)
        # root = self.get_index(1, 0)
        root = self.index
        self.G.add_node(root, thought=self.sys_msg.content, color=COLORMAP["yes"])
        self._grow_thoughts_graph(thoughts, root)
        next_node = next(iter(self.G.neighbors(root)))
        P = self.G.nodes[next_node]["thought"].replace('"', "'")
        state = self.sys_msg, P
        terminate = len(thoughts) <= 1
        return state, terminate

    def _reset(self):
        self.sys_msg = None
        self.thoughts = None
        self.G = nx.DiGraph()
        self.row = 1
        self.col = 0
        self.Gs.append(self.G)
        self.chains = []
        self._last_node = None
        return

    def step(self, action):
        G = self.G
        advice, passed = action
        self.col += 1
        if passed:
            G.nodes[self.index]["color"] = COLORMAP["yes"]
            self._update_sys_on(self.index)
        else:
            previous_index = self.index
            self._color_drop_branch(G, previous_index)
            self.row += 1
            G.add_node(
                self.index,
                thought=f"{self.col}. {advice}",
                color=COLORMAP["comment"],
            )
            G.add_edge(previous_index, self.index, color="whitesmoke")
            self._update_sys_on(self.index)
            resp = self.chat([self.sys_msg])
            chain = self.sys_msg.content + resp.content
            self.chains.append(chain)
            thoughts = self.thoughts_spliter.split(resp.content)
            thoughts = [t for t in thoughts if t != ""]
            self._grow_thoughts_graph(thoughts, self.index)
        next_nodes = list(G.neighbors(self.index))
        if self.col > self.max_steps or len(next_nodes) == 0:
            # TODO: add final request for answer checking
            P = None
            terminate = True
            G.nodes[self.index]["color"] = COLORMAP["done"]
            self.render()
            self.log()
            self.save_line()
        else:
            next_node = next_nodes[0]
            P = G.nodes[next_node]["thought"].replace('"', "'")
            terminate = False
        state = self.sys_msg, P
        return state, terminate

    def log(self):
        for i, chain in enumerate(self.chains):
            logger.info(f"<<<<< CHAIN {i} >>>>>\n" + chain)
        return

    def render(self):
        if not self._render:
            return
        G = self.G
        labels = nx.get_node_attributes(G, "thought")
        colors = nx.get_node_attributes(G, "color").values()
        plt.figure()
        nx.draw(G, labels=labels, node_color=colors, with_labels=True)
        plt.show()
        return

    def save_line(self):
        if self.output is None:
            return
        G_dict = nx.node_link_data(self.G)
        data_json = json.dumps(G_dict)
        with open(self.output, "w+") as fout:
            fout.write(data_json + "\n")
        return

    def save_lines(self):
        if self.output is None:
            return
        logger.info("Saving ...")
        data_jsons = []
        for G in self.Gs:
            G_dict = nx.node_link_data(G)
            data_json = json.dumps(G_dict)
            data_jsons.append(data_json)
        with open(self.output, "w") as fout:
            fout.writelines(data_jsons)
        logger.info(f"Saved data to {self.output}.")
        return

    def get_index(self, row, col):
        return f"T[{row}, {col}]"

    def get_index_loc(self, index):
        row, col = index[2:-1].split(",")
        return int(row.strip()), int(col.strip())

    def _grow_thoughts_graph(self, thoughts, index):
        row, col = self.get_index_loc(index)
        for thought in thoughts:
            col += 1
            previous_index = self.get_index(row, col - 1)
            index = self.get_index(row, col)
            self.G.add_node(
                index,
                thought=f"{col}. {thought}",
                color=COLORMAP["maybe"],
            )
            self.G.add_edge(previous_index, index, color="whitesmoke")
        self._last_node = index
        return

    def _update_sys_on(self, index):
        self.sys_msg = SystemMessage(
            content=self.sys_msg.content + self.G.nodes[index]["thought"] + "\n"
        )
        return

    def _color_drop_branch(self, G, drop_start):
        G.nodes[drop_start]["color"] = COLORMAP["no"]
        drop_nodes = nx.shortest_path(G, drop_start, self._last_node)
        for dnode in drop_nodes[1:]:
            G.nodes[dnode]["color"] = COLORMAP["never"]
        return


class LogiAgent:
    def __init__(self, chat) -> None:
        self.chat = chat
        self.thought_polisher = re.compile(r"^\d+\.\s")
        self._remove_col = lambda x: [t for t in self.thought_polisher.split(x) if t != ""][0]
        # Role maters
        self.PT_prompt_template = HumanMessagePromptTemplate.from_template(PT_prompt)
        self.PF_prompt_template = HumanMessagePromptTemplate.from_template(PF_prompt)
        self.PTF_prompt_template = HumanMessagePromptTemplate.from_template(PTF_prompt)
        self.PF_refined_prompt_template = HumanMessagePromptTemplate.from_template(PF_refined_prompt)

    def policy(self, state):
        sys_msg, P = state
        P = self._remove_col(P)

        PT_msg = self.PT_prompt_template.format(P=P)
        msgs = [sys_msg, PT_msg]
        PT = self.chat(msgs).content

        PF_msg = self.PF_prompt_template.format(P=P)
        msgs = [sys_msg, PF_msg]
        PF = self.chat(msgs).content
        PTF_msg = self.PTF_prompt_template.format(P=P, PT=PT, PF=PF)
        logger.debug("PTF msg:\n" + PTF_msg.content)
        msgs = [sys_msg, PTF_msg]
        choice = self.chat(msgs).content
        logger.debug("choice:\n" + choice)
        if "p is true" in choice.lower():
            advice = P
            passed = True
        elif "p is false" in choice.lower():
            PF_refined_msg = self.PF_refined_prompt_template.format(P=P, PF=PF)
            PF_refined = self.chat([sys_msg, PF_refined_msg]).content
            advice = self._remove_col(PF_refined)
            passed = False
        else:
            logger.warning(choice)
            raise Exception
        return advice, passed
