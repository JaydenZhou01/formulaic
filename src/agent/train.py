import logging
from typing import Optional
from agent.environment import AlphaEnv
from agent.game import create_mcts_player
from agent.network import Network
from sympy import sequence
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from agent.buffer import ReplayBuffer

class AlphaDataset(Dataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        self.rb = replay_buffer

    def __len__(self):
        return len(self.rb)

    def __getitem__(self, idx):
        tr = self.rb.buffer[idx]
        # tr.obs is now a dict {"state": np.ndarray, "stack_len": int}
        state_array = tr.obs["state"]
        stack_len   = tr.obs["stack_len"]

        return {
            'state':     torch.tensor(state_array, dtype=torch.long),  # (T,)
            'stack_len': torch.tensor(stack_len,   dtype=torch.float),# (),
            'pi_target': torch.tensor(tr.pi,       dtype=torch.float),
            'value':     torch.tensor(tr.value,    dtype=torch.float),
            'mask':      torch.tensor(tr.mask,     dtype=torch.bool),
        }

def train_network(
    network: Network,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    num_epochs: int = 10,
    value_coef: float = 1.0,
    l2_reg: float = 1e-4,
    writer: Optional["SummaryWriter"] = None,
    start_step: int = 0,
    logger: Optional["logging.Logger"] = None,
):
    from torch.utils.data import DataLoader

    network.to(device)
    network.train()
    dataset = AlphaDataset(replay_buffer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    global_step = start_step
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch in loader:
            states     = batch['state'].to(device)       # (B, T)
            stack_len  = batch['stack_len'].to(device)   # (B,)
            pi_target  = batch['pi_target'].to(device)   # (B, A)
            value_target  = batch['value'].to(device)       # (B,)
            mask       = batch['mask'].to(device)        # (B, A)

            # forward
            pi_logits, v_logits, v_mean = network(states, stack_len)
            value_pred = v_mean.view(-1)

            # losses
            logp = F.log_softmax(pi_logits, dim=-1)
            logp = logp * mask
            policy_loss = - (pi_target * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, value_target)
            l2_loss = sum(torch.sum(p.pow(2)) for p in network.parameters()) * l2_reg

            loss = policy_loss + value_coef * value_loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if writer:
                writer.add_scalar("Loss/step", loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / len(loader)
        msg = f"[Train] Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

        if writer:
            writer.add_scalar("Loss/epoch", avg_loss, start_step + epoch)

    return global_step


def inference(env:AlphaEnv, model, device, num_games, c_puct_base, c_puct_init, warmup_steps, verbose=True):
    model.eval()
    mcts_player = create_mcts_player(
        network=model,
        device=device,
        num_simulations=50,
        root_noise=False,
        deterministic=False
    )
    total_reward = 0.0
    sequence_set = []
    rewards = []
    for _ in range(num_games):
        env.reset()
        done = False

        root_node = None
        while not done:
            (action, search_pi, root_value, best_child_Q, root_node) = mcts_player(
            env,
            root_node,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            warmup=True if env.steps < warmup_steps else False
        )
            obs, reward, done, _, info = env.step(action)
        if env._tree.is_valid():
            sequence_set.append(env._tree.get_stack())
        else:
            sequence_set.append(env._tokens)
        rewards.append(reward)
        total_reward += reward
    mean_reward = total_reward / num_games

    if verbose:
        for expr, reward in zip(sequence_set, rewards):
            print(f"RPN: {expr}")
            print(f"Reward: {reward}")
        print(f"Mean Reward: {mean_reward:.4f}")
    return mean_reward
