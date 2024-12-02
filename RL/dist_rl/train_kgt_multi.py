import sys

sys.path.insert(0, "../")
import argparse
import torch
import model
import kgtPPO
from dist_ppo import DistPPOProblem
import gym
import sys
import networkx as nx

from pettingzoo.mpe import simple_tag_v2


def main(args):
    steps = 200
    env = simple_tag_v2.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=8,
        max_cycles=steps,
        continuous_actions=True,
    )
    hyperparameters = {
        "timesteps_per_batch": 2000,
        "max_timesteps_per_episode": steps,
        "gamma": 0.99,
        "n_updates_per_iteration": 5,  # epochs
        "lr": 3e-4,
        "clip": 0.2,
        "render": False,
        "render_every_i": 1,
        "save_freq": 10,
        "seed": args.seed,
    }
    env.reset()
    obs_dim = env.observation_spaces["adversary_0"].shape[0]
    act_dim = env.action_spaces["adversary_0"].shape[0]

    base_actor = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
    base_critic = model.FFReLUNet([obs_dim, 64, 64, 64, 1])
    graph = nx.wheel_graph(3)
    dppo = DistPPOProblem(
    base_actor, base_critic, graph, env, **hyperparameters
    )
    kgt_confs = {
    "max_rl_timesteps": 15_000_000,
    "n_updates_per_iteration": 5,
    "alpha_actor": 3e-2,
    "alpha_critic": 3e-3,
    "ID": args.id
    }
    device = torch.device("cpu")

    print("running kgt")
    dopt = kgtPPO.KGTPPO(dppo, device, kgt_confs)
    dopt.train()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', dest='seed', type=int, default=133)             # An int for our seed
    parser.add_argument('--id', dest='id', type=int, default=1)                 # Formal name of environment
    args = parser.parse_args()
    main(args)
