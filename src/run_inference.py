# Evaluate the model from a checkpoint
from html import parser
import os
import argparse
from tabnanny import check
from agent.alpha_pool import AlphaPool
from agent.data import DataContainer
from agent.train import inference
import torch
import torch.nn.functional as F
from agent.network import Network
from agent.environment import AlphaEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--c_puct_base",    type=float, default=100.0)
    parser.add_argument("--c_puct_init",    type=float, default=0.5)
    parser.add_argument("--warmup_steps",   type=int,   default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    initial_rpns = [
        ["close", "dt5", "mean"],
        # ["volume", "log"],
        ["vwap", "close", "sub"]
    ]

    stock_data = DataContainer(device=device)
    pool = AlphaPool(initial_rpns, stock_data)
    env = AlphaEnv(pool, device, False)

    checkpoint = torch.load(args.model_path, map_location=device)
    net = Network(env.action_size, 128)
    net.load_state_dict(checkpoint['model_state'])
    net.to(device)
    net.eval()

    avg_r = inference(env, net, device, args.num_games, args.c_puct_base, args.c_puct_init, args.warmup_steps)
    print(f"Average reward over {args.num_games} games: {avg_r:.4f}")

if __name__ == "__main__":
    main()

