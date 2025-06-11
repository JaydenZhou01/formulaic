import copy
from agent.environment import AlphaEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_W = defaultdict(float)
        self.child_N = defaultdict(float)

class Node:
    """
    Node in the MCTS tree
    """

    def __init__(
            self,
            num_actions,
            action=None,
            parent=None,
    ):
        self.num_actions = num_actions
        self.action = action
        self.parent = parent
        self.is_expanded = False

        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_P = np.zeros(num_actions, dtype=np.float32)

        self.children = {}


    def child_U(self, c_puct_base, c_puct_init):
        pb_c = math.log((1.0 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))
    
    def child_Q(self):
        child_N = np.where(self.child_N > 0, self.child_N, 1)
        return self.child_W / child_N
    
    @property
    def N(self):
        return self.parent.child_N[self.action]
    
    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        return self.parent.child_W[self.action]
    
    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        if self.parent.child_N[self.action] > 0:
            return self.parent.child_W[self.action] / self.parent.child_N[self.action]  
        else:
            return 0.0
        
    @property
    def has_parent(self):
        return isinstance(self.parent, Node)
    
def select_child(
        node: Node,
        legal_actions,
        c_puct_base,
        c_puct_init
):
    """
    Select the child with the highest UCB value
    """
    assert node.is_expanded

    ucb_scores = node.child_Q() + node.child_U(c_puct_base, c_puct_init)
    ucb_scores = np.where(legal_actions == 1, ucb_scores, -np.inf)

    action = int(np.argmax(ucb_scores))

    if action not in node.children:
        node.children[action] = Node(num_actions=node.num_actions, action=action, parent=node)

    return node.children[action]

def expand(node: Node, prior_prob):
    assert not node.is_expanded
    node.child_P = prior_prob
    node.is_expanded = True

def backup(node: Node, value):
    while isinstance(node, Node):
        node.N += 1
        node.W += value
        node = node.parent

def add_dirichlet_noise(node: Node, legal_actions, eps=0.25, alpha=0.03):
    alphas = np.ones_like(legal_actions) * alpha
    noise = legal_actions * np.random.dirichlet(alphas)
    node.child_P = (1 - eps) * node.child_P + eps * noise

def generate_search_policy(child_N, temparature, legal_actions):
    child_N = legal_actions * child_N

    if temparature > 0.0:
        exp = max(1.0, min(5.0, 1.0 / temparature))
        child_N = np.power(child_N, exp)
    
    pi_probs = child_N / np.sum(child_N)
    return pi_probs

def mcts_search(
        env: AlphaEnv,
        eval_func,
        root_node,
        c_puct_base,
        c_puct_init,
        num_simulations,
        root_noise,
        warm_up,
        deterministic,
):
    
    if root_node is None:
        prior_prob, value_logits, value_mean = eval_func(env.observation(), False)
        root_node = Node(num_actions=env.action_size, parent=DummyNode())
        expand(root_node, prior_prob)
        backup(root_node, value_mean)

    root_legal_actions = env.legal_actions
    if root_noise:
        add_dirichlet_noise(root_node, root_legal_actions)
    
    sim_env = copy.deepcopy(env)
    env_state = {"steps": sim_env.steps,
                 "state": sim_env.state,
                 "tokens": sim_env._tokens,
                 "tree": sim_env._tree}
    
    while root_node.N < num_simulations:
        node = root_node
        
        # sim_env = copy.deepcopy(env)
        sim_env.reset_to(copy.deepcopy(env_state))

        obs = sim_env.observation()
        done = False

        # Selection
        while node.is_expanded:
            node = select_child(
                node,
                sim_env.legal_actions,
                c_puct_base,
                c_puct_init
            )
            obs, reward, done, _, _ = sim_env.step(node.action)
            if done:
                break

        if done:
            backup(node, reward)
            continue

        # Expansion
        obs = sim_env.observation()
        prior_prob, value_prob, value_mean = eval_func(obs, False)
        expand(node, prior_prob)

        # Backup
        backup(node, value_mean)

    search_pi = generate_search_policy(root_node.child_N, 1.0 if warm_up else 0.1, root_legal_actions)

    action = None
    next_root_node = None
    best_child_Q = 0.0

    if deterministic:
        action = int(np.argmax(root_node.child_N))
    else:
        action = int(np.random.choice(root_node.num_actions, p=search_pi))

    if action in root_node.children:
        next_root_node = root_node.children[action]
        
        N, W = copy.copy(next_root_node.N), copy.copy(next_root_node.W)
        next_root_node.parent = DummyNode()
        next_root_node.N, next_root_node.W = N, W
        next_root_node.action = None

        best_child_Q = next_root_node.Q

    return (action, search_pi, root_node.Q, best_child_Q, next_root_node)

