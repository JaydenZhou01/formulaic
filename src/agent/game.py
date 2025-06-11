import json
import random
from agent.buffer import ReplayBuffer, Transition
from agent.config import REWARD_THRESHOLD
from agent.environment import AlphaEnv
from sympy import root
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from typing import List, Dict, Callable
from agent.mcts import Node, mcts_search



def create_mcts_player(
        network: nn.Module,
        device: torch.device,
        num_simulations: int,
        root_noise: bool=True,
        deterministic: bool=False,
):
    @torch.no_grad()
    def eval_position(
        state,
        batched: bool=False,
    ):
        
        # 1) unify to a list of dicts
        if not batched:
            states = [state]
        else:
            states = state  # already a list of dicts

        B = len(states)
        # 2) extract token arrays and stack lengths
        token_seqs = np.stack([s["state"] for s in states], axis=0)        # (B, T)
        stack_lens = np.array([s["stack_len"] for s in states], dtype=np.int32)  # (B,)

        # 3) to torch
        tokens = torch.from_numpy(token_seqs).long().to(device)       # [B, T]
        slens  = torch.from_numpy(stack_lens).float().to(device)     # [B]

        # 4) forward through modified network
        #    which now expects (tokens, stack_len)
        pi_logits, v_logits, v_mean = network(tokens, slens)

        pi_probs = F.softmax(pi_logits, dim=-1).cpu().numpy()
        v_logprobs = F.log_softmax(v_logits, dim=-1).cpu().numpy()

        pis = [pi_probs[i] for i in range(B)]
        vlogs = [v_logprobs[i] for i in range(B)]

        if not batched:
            return pis[0], vlogs[0], v_mean[0].item()
        return pis, vlogs, v_mean
    
    def actor(
        env,
        root_node,
        c_puct_base,
        c_puct_init,
        warmup=False
    ):
        return mcts_search(
            env=env,
            eval_func=eval_position,
            root_node=root_node,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            num_simulations=num_simulations,
            root_noise=root_noise,
            warm_up=warmup,
            deterministic=deterministic,
        )

    return actor

def selfplay(
    seed: int,
    network: nn.Module,
    device: torch.device,
    env: AlphaEnv,
    num_simulations: int,
    c_puct_base: float,
    c_puct_init: float,
    warmup_steps: int,
    num_games: int,
    logger
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    network = network.to(device)
    network.eval()

    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        root_noise=True,
        deterministic=False,
    )

    replay = ReplayBuffer(capacity=10000, seed=seed)
    avg_reward = 0.0
    for _ in range(num_games):
        game_seq = play_game(
            env=env,
            mcst_player=mcts_player,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            num_simulations=num_simulations,
            root_noise=True,
            warmup_steps=warmup_steps,
            deterministic=False,
        )

        # print("Game Finished.")
        # print game sequence
        # for i, transition in enumerate(game_seq):
        #     print(f"Step {i}: State: {transition.state}, Pi: {transition.pi_probs}, Value: {transition.value}")
        # env.render()
        # print("Reward: ", game_seq[-1].value)
        if env._tree.is_valid():
            logger.info("Expression: %s", env._tree.get_stack())
        else:
            logger.info("Invalid expression: %s", env._tokens)
        reward = game_seq[-1].value
        # outlier check
        if reward > 0.5:
            breakpoint()
        if reward >= REWARD_THRESHOLD:
            # record the expression and reward into json file
            logger.info("Valid expression found: %s", env._tree.get_stack())
            with open("valid_expressions.json", "a") as f:
                json.dump({
                    "expression": str(env._tree.get_stack()),
                    "reward": reward
                }, f)
                f.write("\n")
        logger.info("Reward: %s", reward)

        avg_reward += reward
        replay.add_game(game_seq)
    
    avg_reward /= num_games
    return replay, avg_reward



def play_game(
    env: AlphaEnv,
    mcst_player,
    c_puct_base,
    c_puct_init,
    num_simulations,
    root_noise,
    warmup_steps,
    deterministic,
    ):
    
    obs, info = env.reset()
    done = False
    episode : List[Transition] = []
    root_node = None

    step_rewards = []
    observations = []
    pis = []
    masks = []
    while not done:
        (action, search_pi, root_value, best_child_Q, root_node) = mcst_player(
            env,
            root_node,
            c_puct_base,
            c_puct_init,
            warmup=True if env.steps < warmup_steps else False
        )

        obs, reward, done, _, info = env.step(action)
        mask = info.get('action_masks')

        observations.append(obs.copy())
        pis.append(search_pi.copy())
        masks.append(mask.copy())
        step_rewards.append(reward)

    returns = []
    R = 0.0
    for r in reversed(step_rewards):
        R = r + 0.99 * R
        returns.append(R)
    returns = list(reversed(returns))

    for (obs, pi, value, mask) in zip(observations, pis, returns, masks):
        episode.append(Transition(
            obs=obs,
            pi=pi,
            value=value,
            mask=mask
        ))
    return episode
        
