import re
import json
from loguru import logger
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
import networkx as nx
import matplotlib.pyplot as plt
from .prompts import (
    ARGUE,
    NEGATION,
    SYS_PROMPTS,
    ANSWER_PROMPTS,
)
from .utils import load_data, extract_pred_answer, INVALID_ANS

COLORMAP = {
    "maybe": "yellow",
    "yes": "green",
    "no": "red",
    "comment": "lightgreen",
    "never": "gray",
    "done": "purple",
    "edge": "gray",
}


class ThoughtEnv:
    def __init__(
        self,
        chat,
        max_steps=25,
        output=None,
        dataset_name="GSM8K",
        prompt_version=0,
        debug=False,
    ) -> None:
        self.chat = chat
        self.max_steps = max_steps
        self.thoughts_spliter = re.compile(r"(?:\n+|^)#\d+\.\s")
        self.sys_template = SystemMessagePromptTemplate.from_template(
            SYS_PROMPTS[dataset_name][prompt_version]
        )
        self.answer_msg = HumanMessage(content=ANSWER_PROMPTS[dataset_name])
        self._debug = debug
        if debug:
            logger.level("DEBUG")
            logger.add("logs/debug/file_{time}.log")
        else:
            logger.level("INFO")
            logger.add("logs/release/file_{time}.log")
        self.output = output
        self.datas = load_data(self.output, format="dict")
        self.dataset_name = dataset_name
        self.index_node = None

    def reset(self, data):
        terminate = self._reset(data)
        if terminate:
            state = None, None, None, False
            return state, terminate
        init_sys_msg = SystemMessage(content=self.sys_msg.content + "#1. ")
        resp = self.chat([init_sys_msg])
        chain = init_sys_msg.content + resp.content
        logger.debug("<< CHAIN 1 >>\n" + chain)
        self.default_chain = SystemMessage(content=chain)
        thoughts = self.thoughts_spliter.split(resp.content)
        root = self.get_index(1, 0)
        self.G.add_node(
            root,
            thought=self.sys_msg.content,
            color=COLORMAP["yes"],
            refined=True,
        )
        self._grow_thoughts_graph(thoughts, root)
        next_node = next(iter(self.G.neighbors(root)))
        self.index_node = next_node
        P = self.G.nodes[self.index_node]["thought"].replace('"', "'")
        refined = self.G.nodes[self.index_node]["refined"]
        state = self.sys_msg, 1, P, refined
        terminate = len(thoughts) < 1
        return state, terminate

    def _reset(self, data):
        terminate = False
        idx = data["idx"]
        if idx in self.datas:
            logger.warning(
                f"Loading existing data from {self.output} with idx {idx}. Terminate early."
            )
            data = self.datas[idx]
            self.G = nx.node_link_graph(data["G"])
            self.data = data
            terminate = True
            return terminate
        self.G = nx.DiGraph()
        self._last_node = None
        self.sys_msg = self.sys_template.format(question=data["question"])
        self.default_chain = None
        self.data = data
        logger.info(f"Reset for question {idx}.")
        return terminate

    def get_next_node(self):
        next_nodes = list(self.G.neighbors(self.index_node))
        if len(next_nodes) == 0:
            next_node = None
        else:
            next_node = next_nodes[0]
        return next_node

    def step(self, action):
        terminate = False
        G = self.G
        advice, passed = action
        row, col = self.get_index_loc(self.index_node)
        if passed:
            G.nodes[self.index_node]["color"] = COLORMAP["yes"]
            self._update_sys_on(self.index_node)
            next_node = self.get_next_node()
        else:
            self._color_drop_branch(G, self.index_node)
            next_node = self.get_index(row + 1, col)
            G.add_node(
                next_node,
                thought=f"#{col}. {advice}\n",
                color=COLORMAP["comment"],
                refined=True,
            )
            G.add_edge(self.index_node, next_node, color=COLORMAP["edge"])
            new_sys_msg = SystemMessage(
                content=self.sys_msg.content + self.G.nodes[next_node]["thought"] + "\n"
            )
            resp = self.chat([new_sys_msg])
            chain = (
                self.sys_msg.content + self.G.nodes[next_node]["thought"] + resp.content
            )
            logger.debug(f"<< CHAIN {row + 1} >>\n" + chain)
            thoughts = self.thoughts_spliter.split(resp.content)
            thoughts = [t for t in thoughts if t != ""]
            self._grow_thoughts_graph(thoughts, next_node)
        if col > self.max_steps or next_node is None:
            # TODO: if exceeds maxstep, just return the initial
            terminate = True
            self._terminate()
            return None, terminate
        self.index_node = next_node
        P = G.nodes[self.index_node]["thought"].replace('"', "'")
        refined = G.nodes[self.index_node]["refined"]
        _, col = self.get_index_loc(self.index_node)
        state = self.sys_msg, col, P, refined
        return state, terminate

    def _terminate(self):
        self.G.nodes[self.index_node]["color"] = COLORMAP["done"]
        self._store_graph_callback()
        self._compute_answer_callback()
        self._save_data_callback()
        self.render()
        return

    def _store_graph_callback(self):
        G_dict = nx.node_link_data(self.G)
        self.data["G"] = G_dict
        return

    def _compute_answer_callback(self):
        recall = 0
        answer_default = self._extract_answer(chain=self.default_chain)
        answer_logi = self._extract_answer(chain=self.sys_msg)
        if answer_logi == INVALID_ANS:
            # Adopt konwn eaiser thought answer
            answer_logi = answer_default
            recall = 1
        self.data.update(
            {
                "answer_default": answer_default,
                "answer_logi": answer_logi,
                "recall": recall,
            }
        )
        return

    def _extract_answer(self, chain):
        answer_pred = self.chat([chain, self.answer_msg]).content
        logger.debug(answer_pred)
        answer_pred = extract_pred_answer(
            dataset_name=self.dataset_name, pred_completion=answer_pred
        )
        return answer_pred

    def _save_data_callback(self):
        if self.output is None:
            return
        with open(self.output, "a+") as fout:
            data_json = json.dumps(self.data)
            fout.write(data_json + "\n")
            logger.info(f"Saved one line to {self.output}")
        return

    def render(self):
        if not self._debug:
            return
        G = self.G
        labels = nx.get_node_attributes(G, "thought")
        colors = nx.get_node_attributes(G, "color").values()
        plt.figure()
        nx.draw(G, labels=labels, node_color=colors, with_labels=True)
        plt.show()
        return

    def get_index(self, row, col):
        return f"T[{row}, {col}]"

    def get_index_loc(self, index):
        row, col = index[2:-1].split(",")
        return int(row.strip()), int(col.strip())

    def _grow_thoughts_graph(self, thoughts, index):
        row, col = self.get_index_loc(index)
        for thought in thoughts:
            thought = thought.strip("\n")
            thought = thought.strip("<step> ")
            thought = thought.strip(" </step>")
            col += 1
            previous_index = self.get_index(row, col - 1)
            index = self.get_index(row, col)
            self.G.add_node(
                index,
                thought=f"#{col}. {thought}",
                color=COLORMAP["maybe"],
                refined=False,
            )
            self.G.add_edge(previous_index, index)
        self._last_node = index
        # self.render()
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
    def __init__(self, chat, check_refined=True, mode="argue_review") -> None:
        self.chat = chat
        self._mode = mode  # ['negation', 'argue_review', 'argue_noreview']
        self.thought_polisher = re.compile(r"#\d+[:\.]\s")
        self._remove_col = lambda x: [
            t for t in self.thought_polisher.split(x) if len(t) > 2
        ][-1]
        self.check_refined = check_refined

    def policy(self, state):
        sys_msg, col, P, refined = state
        P = self._remove_col(P)
        if refined and not self.check_refined:
            return P, True
        if self._mode == 'naive':
            return P, True
        if self._mode.startswith("argue"):
            # Role maters
            self.PT_template = HumanMessagePromptTemplate.from_template(ARGUE.PT)
            self.PF_template = HumanMessagePromptTemplate.from_template(ARGUE.PF)
            self.PJ_template = HumanMessagePromptTemplate.from_template(ARGUE.PJ)
            PT_msg = self.PT_template.format(P=P, col=col)
            msgs = [sys_msg, PT_msg]
            # logger.debug(sys_msg.content + PT_msg.content)
            PT = self.chat(msgs).content

            PF_msg = self.PF_template.format(P=P, col=col)
            msgs = [sys_msg, PF_msg]
            # logger.debug(sys_msg.content + PF_msg.content)
            PF = self.chat(msgs).content
            PJ_msg = self.PJ_template.format(P=P, PT=PT, PF=PF, col=col)
            msgs = [sys_msg, PJ_msg]
            choice = self.chat(msgs).content
            logger.debug("PTF msg:\n" + sys_msg.content + PJ_msg.content)
            logger.debug("Choice:\n" + choice)
            if " is true" in choice.lower():
                advice = P
                passed = True
            elif " is false" in choice.lower():
                if self._mode == "argue_noreview":
                    self.PR_template = HumanMessagePromptTemplate.from_template(
                        ARGUE.PR_N
                    )
                    PFR_msg = self.PR_template.format(P=P, col=col)
                else:
                    self.PR_template = HumanMessagePromptTemplate.from_template(
                        ARGUE.PR
                    )
                    PFR_msg = self.PR_template.format(P=P, PF=PF, col=col)
                logger.debug(sys_msg.content + PFR_msg.content)
                PFR = self.chat([sys_msg, PFR_msg]).content
                advice = self._remove_col(PFR)
                logger.info("Refinement:\n" + advice)
                passed = False
            else:
                logger.warning(choice)
                advice = P
                passed = True
        elif self._mode.startswith("negation"):
            self.PF_template = HumanMessagePromptTemplate.from_template(NEGATION.PF)
            self.PR_template = HumanMessagePromptTemplate.from_template(NEGATION.PR)
            PF_msg = self.PF_template.format(P=P, col=col)
            msgs = [sys_msg, PF_msg]
            # logger.debug(sys_msg.content + PF_msg.content)
            PF = self.chat(msgs).content
            if ' is true' in PF.lower():
                advice = P
                passed = True
            else:
                PR_msg = self.PR_template.format(P=P, PF=PF, col=col)
                msgs = [sys_msg, PR_msg]
                revision = self.chat(msgs).content
                logger.debug("Revision:\n" + revision)
                passed = False
                advice = revision
        return advice, passed
