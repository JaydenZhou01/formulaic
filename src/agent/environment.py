from cmath import isnan
import math
from typing import Optional

from agent.expr_tree import ExprTree
from agent.alpha_pool import AlphaPool
from agent.expression import *
from agent.config import *
from agent.tokens import *

import numpy as np
import gymnasium as gym
import torch


class AlphaEnv(gym.Env):
    def __init__(self, pool:AlphaPool, device, verbose=False):
        self.max_expr_length = MAX_EXPR_LENGTH
        self.action_size = len(OPERATORS) + len(FeatureType) + len(DELTA_TIMES) + len(CONSTANTS) + len(SeparatorType) - 1
        self._actions = (
            [OperatorToken(op) for op in OPERATORS] +
            [FeatureToken(f) for f in FeatureType] +
            [DeltaTimeToken(d) for d in DELTA_TIMES] +
            [ConstantToken(c) for c in CONSTANTS] +
            [SEP_TOKEN]
        )
        self.action_space = gym.spaces.Discrete(self.action_size)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=0, high=self.action_size, shape=(self.max_expr_length,), dtype=np.uint8),
            "stack_len": gym.spaces.Box(low=0, high=self.max_expr_length, shape=(), dtype=np.int32)})
        self.eval_cnt = 0
        self.steps = 0
        self.pool = pool
        self._device = device
        self._verbose = verbose

        self.render_mode = None
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self._tokens: List[Token] = [BEG_TOKEN]
        self._tree = ExprTree()
        self.legal_actions = self.action_masks()
        return self.observation(), {'action_masks': self.legal_actions, 'stack_height': 0}
    
    def reset_to(self, env_state: dict):
        self.steps = env_state['steps']
        self.state = env_state['state']
        self._tokens = env_state['tokens']
        self._tree = env_state['tree']
        self.legal_actions = self.action_masks()

        
    def step(self, action: int):
        action_token = self._actions[action]
        reward = -COST_PER_STEP
        done = False
        if (isinstance(action_token, SeparatorToken) and
                action_token.indicator == SeparatorType.SEP and
                self.steps > 0):
            if self.steps >= MIN_EXPR_LENGTH:
                reward += self._get_reward()
            else:
                reward += -1.0
            done = True
            self._tokens.append(action_token)
            self.state[self.steps] = action
        elif len(self._tokens) < self.max_expr_length:
            self._tokens.append(action_token)
            self._tree.push(action_token)
        else:
            done = True
            reward += self._get_reward() if self._tree.is_valid() else -1.0
        if math.isnan(reward):
            reward = 0.
        if not done:
            self.state[self.steps] = action
            self.steps += 1
            self.legal_actions = self.action_masks()
        return self.observation(), reward, done, False, {'action_masks': self.legal_actions}
    
    def observation(self):
        tokens = self.state.copy()
        stack_len = len(self._tree.stack)
        return {
            "state": tokens,
            "stack_len": stack_len
        }

    def _get_reward(self):
        expr = self._tree.get_stack()
        if self._verbose:
            print(expr)
        reward = self.pool.compute_marginal_contribution(expr)
        self.eval_cnt += 1
        return reward + TERMINATION_BONUS
    
    def _token2id(self, token: Token):
        for i, t in enumerate(self._actions):
            if isinstance(token, t.__class__) and str(token)==str(t):
                return i
        return -1
    
    def action_masks(self):
        mask = np.zeros(len(self._actions), dtype=np.uint8)
        for i, token in enumerate(self._actions):
            if isinstance(token, SeparatorToken):
                if token.indicator == SeparatorType.SEP:
                    mask[i] = self._tree.is_valid()
                else:
                    mask[i] = 0
            else:
                if self._tree.validate(token):
                    mask[i] = 1
        return mask
    
    def render(self, mode='human'):
        if mode == 'human':
            if self._tree.is_valid():
                print(f"Expression: {self._tree.get_stack()}")
            else:   
                print(self._tokens)
        else:
            raise NotImplementedError("Render mode not supported.")