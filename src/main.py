import agent, importlib
from agent.environment import AlphaEnv
from agent.game import selfplay
from agent.network import Network
import torch

env = AlphaEnv()
net = Network(7, 128)
buffer = selfplay(42, net, torch.device('cpu'), env, 100, 20000, 1.0, 16, 10)