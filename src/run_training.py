import os
import argparse
import logging

from agent.alpha_pool import AlphaPool
from agent.data import DataContainer
from agent.train import inference
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.environment import AlphaEnv
from agent.game import selfplay
from agent.network import Network
from agent.train import train_network  # your train_network function

def parse_args():
    p = argparse.ArgumentParser("AlphaZero-style self-play + training")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--num_iterations", type=int,   default=50,
                   help="Number of self-play / training cycles")
    p.add_argument("--num_games",      type=int,   default=20,
                   help="Self-play games per iteration")
    p.add_argument("--num_sims",       type=int,   default=50,
                   help="MCTS simulations per move")
    p.add_argument("--c_puct_base",    type=float, default=1000.0)
    p.add_argument("--c_puct_init",    type=float, default=1.0)
    p.add_argument("--warmup_steps",   type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--num_epochs",     type=int,   default=50,
                   help="Training epochs per iteration")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--save_every",     type=int,   default=5,
                   help="Save checkpoint every N iterations")
    p.add_argument("--log_dir",        type=str,   default="logs")
    p.add_argument("--ckpt_dir",       type=str,   default="checkpoints")
    p.add_argument("--load_ckpt",      type=str,   default=None,
                   help="Path to checkpoint to load")
    return p.parse_args()

def main():
    args = parse_args()

    # prepare dirs
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # setup logging
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # file handler
    file_path = os.path.join(args.log_dir, "training.log")
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("AlphaZero")
    logger.info(f"Logging to console and {file_path}")

    # TensorBoard
    args.log_dir = os.path.join(args.log_dir, "version_1")
    writer = SummaryWriter(log_dir=args.log_dir)

    # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    initial_rpns = [
        ["close", "dt5", "mean"],
        # ["volume", "log"],
        ["vwap", "close", "sub"]
    ]

    stock_data = DataContainer(device=torch.device("cpu"))
    pool = AlphaPool(initial_rpns, stock_data)
    env = AlphaEnv(pool, device, False)
    net = Network(env.action_size, 128)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.load_ckpt:
        logger.info(f"Loading checkpoint from {args.load_ckpt}")
        checkpoint = torch.load(args.load_ckpt, map_location=device)
        net.load_state_dict(checkpoint["model_state"])
        net.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = checkpoint["iteration"] + 1
    else:
        start_iter = 1
        net.to(device)
        net.train()
        logger.info("Training from scratch")

    global_step = 0
    best_reward = -float("inf")
    for it in range(start_iter, args.num_iterations + 1):
        logger.info(f"=== Iteration {it} ===")

        # --- self-play to collect data ---
        logger.info("Self-play: generating games...")
        buffer, avg_reward = selfplay(
            seed=args.seed + it,
            network=net,
            device=device,
            env=env,
            num_simulations=args.num_sims,
            c_puct_base=args.c_puct_base,
            c_puct_init=args.c_puct_init,
            warmup_steps=args.warmup_steps,
            num_games=args.num_games,
            logger=logger
        )
        logger.info(f"Collected {len(buffer)} transitions")
        logger.info(f"Average reward from self-play: {avg_reward:.4f}")
        writer.add_scalar("Reward/selfplay", avg_reward, it)

        # --- train network ---
        logger.info("Training network...")
        global_step = train_network(
            network=net,
            replay_buffer=buffer,
            optimizer=optimizer,
            batch_size=args.batch_size,
            device=device,
            num_epochs=args.num_epochs,
            writer=writer,
            start_step=global_step,
            logger=logger,
        )

        # --- optional evaluation ---
        # if it % args.eval_every == 0:
        #     logger.info("Running evaluation...")
        #     rewards = []
        #     avg_r = inference(
        #         env=env,
        #         model=net,
        #         device=device,
        #         num_games=args.num_games,
        #         c_puct_base=args.c_puct_base,
        #         c_puct_init=args.c_puct_init,
        #         warmup_steps=args.warmup_steps,
        #         verbose=True
        #     )
        #     logger.info(f"[Eval] Mean Reward = {avg_r:.4f}")
        #     writer.add_scalar("Reward/eval", avg_r, it)

        # --- save checkpoint ---
        if it % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"iter_{it:03d}.pt")
            torch.save({
                "iteration": it,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        # save best model
        if it == 1 or avg_reward > best_reward:
            best_reward = avg_reward
            best_model_path = os.path.join(args.ckpt_dir, "best_model.pt")
            torch.save({
                "iteration": it,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")

    writer.close()

if __name__ == "__main__":
    main()