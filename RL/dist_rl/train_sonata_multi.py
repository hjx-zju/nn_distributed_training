import sys

sys.path.insert(0, "../")
import argparse
import torch
import model
import sonataPPO
from dist_ppo import DistPPOProblem
import gym
import sys
import networkx as nx

from pettingzoo.mpe import simple_tag_v2


def main(args):
    for ii in range(1):
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
            "lr": 3e-3,
            "clip": 0.2,
            "render": False,
            "render_every_i": 1,
            "save_freq": 100,
            "seed":args.seed,
        }
        env.reset()
        obs_dim = env.observation_spaces["adversary_0"].shape[0]
        act_dim = env.action_spaces["adversary_0"].shape[0]

        base_actor = model.FFReLUNet([obs_dim, 64, 64, 64, act_dim])
        base_critic = model.FFReLUNet([obs_dim, 64, 64, 64, 1])
        graph = nx.wheel_graph(3)
        dppo = DistPPOProblem(base_actor, base_critic, graph, env, **hyperparameters)
        sonata_confs = {
            "tau": 0.0001,
            "alpha": 1,
            "primal_lr_start": 5e-4,
            "primal_lr_finish": 5e-5,
            "critic_lr_start": 5e-4,
            "critic_lr_finish": 5e-5,
            "lr_decay_type": "linear",
            "critic_lr_decay_type": "linear",
            "persistant_primal_opt": False,
            "primal_iterations": hyperparameters["n_updates_per_iteration"],
            "max_rl_timesteps": 15_000_000,
            "outer_iterations": 15_000_000,
            "ID": args.id,
        }
        device = torch.device("cpu")

        print("running sonata")
        dopt = sonataPPO.SONATAPPO(dppo, device, sonata_confs)
        dopt.train()


if __name__ == "__main__":
    # Parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--seed', dest='seed', type=int, default=133)             # An int for our seed
	parser.add_argument('--id', dest='id', type=int, default=5)                 # Formal name of environment

	args = parser.parse_args()

	# Collect data
	main(args)
