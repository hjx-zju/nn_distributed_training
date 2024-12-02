import torch
from utils import graph_generation
import copy
import numpy as np


class KGTPPO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all actor parameter pointers
        self.plists_actor = {
            i: list(self.pr.actors[i].parameters()) for i in range(self.pr.N)
        }
        self.num_params_actor = len(self.plists_actor[0])
        base_zeros_actor = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists_actor[0]
        ]
        self.glists_actor = {
            i: copy.deepcopy(base_zeros_actor) for i in range(self.pr.N)
        }
        self.ylists_actor = {
            i: copy.deepcopy(base_zeros_actor) for i in range(self.pr.N)
        }
        self.clists_actor = {
            i: copy.deepcopy(base_zeros_actor) for i in range(self.pr.N)
        }
        # Get list of all critic parameter pointers
        self.plists_critic = {
            i: list(self.pr.critics[i].parameters()) for i in range(self.pr.N)
        }
        # Useful numbers
        self.num_params_critic = len(self.plists_critic[0])
        base_zeros_critic = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists_critic[0]
        ]
        self.glists_critic = {
            i: copy.deepcopy(base_zeros_critic) for i in range(self.pr.N)
        }
        self.ylists_critic = {
            i: copy.deepcopy(base_zeros_critic) for i in range(self.pr.N)
        }
        self.clists_critic = {
            i: copy.deepcopy(base_zeros_critic) for i in range(self.pr.N)
        }
        # Training hyper params
        self.alpha_actor = conf["alpha_actor"]
        self.alpha_critic = conf["alpha_critic"]

    def train(self, profiler=None):
        # eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        max_rl_timesteps = self.conf["max_rl_timesteps"]

        # Comm weights
        W = graph_generation.get_metropolis(self.pr.graph)
        W = W.to(self.device)

        # Initialize Ylists and Glists
        self.pr.split_rollout_marl()
        self.pr.update_advantage()
        for i in range(self.pr.N):
            # This requires one rollout but no steps are taken yet
            actor_loss, critic_loss = self.pr.ev_ppo_loss(i)
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.pr.actors[i].parameters(), 0.5)
            
            with torch.no_grad():
                for p in range(self.num_params_actor):
                    self.ylists_actor[i][p] = (
                        -self.plists_actor[i][p].grad.detach().clone()
                    )
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pr.critics[i].parameters(), 0.5)
            
            with torch.no_grad():
                for p in range(self.num_params_critic):
                    self.ylists_critic[i][p] = (
                        -self.plists_critic[i][p].grad.detach().clone()
                    )
        for i in range(self.pr.N):
            with torch.no_grad():
                neighs = list(self.pr.graph.neighbors(i))
                num_neighs = len(neighs) + 1
                for p in range(self.num_params_actor):
                    self.ylists_actor[i][p] += (
                        self.plists_actor[i][p].grad.detach().clone() / num_neighs
                    )
                    for j in neighs:
                        self.ylists_actor[i][p] += (
                            self.plists_actor[j][p].grad.detach().clone() / num_neighs
                        )
       
        # Optimization loop
        k = 0
        avg_ep_rews = []
        timesteps = []
        avg_loss = []
        agree_0 = np.array([])
        agree_1 = np.array([])
        agree_2 = np.array([])
        while self.pr.logger["t_so_far"] < max_rl_timesteps:
            # Compute graph weights
            # Iterate over the agents for communication step
            self.pr.split_rollout_marl()
            self.pr.update_advantage()
            bak_plists_actor = copy.deepcopy(self.plists_actor)
            bak_plists_critic = copy.deepcopy(self.plists_critic)
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                
                for _ in range(self.conf["n_updates_per_iteration"]):
                    actor_loss, critic_loss = self.pr.ev_ppo_loss(i)
                    self.pr.actors[i].zero_grad()
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.pr.actors[i].parameters(), 0.5)
                    with torch.no_grad():
                        for p in range(self.num_params_actor):
                            self.plists_actor[i][p].add_(
                                self.plists_actor[i][p].grad, alpha=-self.alpha_actor
                            )
                            self.plists_actor[i][p].add_(
                                self.clists_actor[i][p], alpha=-self.alpha_actor
                            )
                    self.pr.critics[i].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.pr.critics[i].parameters(), 0.5)
                    
                    with torch.no_grad():
                        for p in range(self.num_params_critic):
                            self.plists_critic[i][p].add_(
                                self.plists_critic[i][p].grad, alpha=-self.alpha_critic
                            )
                            self.plists_critic[i][p].add_(
                                self.clists_critic[i][p], alpha=-self.alpha_critic
                            )
            for i in range(self.pr.N):
                with torch.no_grad():
                    for p in range(self.num_params_actor):
                        self.ylists_actor[i][p].zero_()
                        self.ylists_actor[i][p].add_(
                            bak_plists_actor[i][p], alpha=1 / self.conf["n_updates_per_iteration"] / self.alpha_actor
                        )
                        self.ylists_actor[i][p].add_(
                            self.plists_actor[i][p], alpha=-1 / self.conf["n_updates_per_iteration"] / self.alpha_actor
                        )
                    for p in range(self.num_params_critic):
                        self.ylists_critic[i][p].zero_()
                        self.ylists_critic[i][p].add_(
                            bak_plists_critic[i][p], alpha=1 / self.conf["n_updates_per_iteration"] / self.alpha_critic
                        )
                        self.ylists_critic[i][p].add_(
                            self.plists_critic[i][p], alpha=-1 / self.conf["n_updates_per_iteration"] / self.alpha_critic
                        )
            
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    for p in range(self.num_params_actor):
                        self.clists_actor[i][p].add_(self.ylists_actor[i][p], alpha=W[i, i]-1)
                        self.plists_actor[i][p].set_(W[i, i] * (bak_plists_actor[i][p] - self.conf["n_updates_per_iteration"] * self.alpha_actor * self.ylists_actor[i][p]))
                        for j in neighs:
                            self.clists_actor[i][p].add_(self.ylists_actor[j][p], alpha=W[i, j])
                            self.plists_actor[i][p].add_(bak_plists_actor[j][p] - self.conf["n_updates_per_iteration"] * self.alpha_actor * self.ylists_actor[j][p], alpha=W[i, j])
                    for p in range(self.num_params_critic):
                        self.clists_critic[i][p].add_(self.ylists_critic[i][p], alpha=W[i, i]-1)
                        self.plists_critic[i][p].set_(W[i, i] * (bak_plists_critic[i][p] - self.conf["n_updates_per_iteration"] * self.alpha_critic * self.ylists_critic[i][p]))
                        for j in neighs:
                            self.clists_critic[i][p].add_(self.ylists_critic[j][p], alpha=W[i, j])
                            self.plists_critic[i][p].add_(bak_plists_critic[j][p] - self.conf["n_updates_per_iteration"] * self.alpha_critic * self.ylists_critic[j][p], alpha=W[i, j])

            avg_loss.append(
                [
                    np.mean(
                        [
                            losses.float().mean()
                            for losses in self.pr.logger["actor_losses"]
                        ]
                    ),
                    np.mean(
                        [
                            losses.float().mean()
                            for losses in self.pr.logger["critic_losses"]
                        ]
                    ),
                ]
            )
            if k % 10 == 0:
                np.save(
                    f'./trained/avg_loss_kgt_{self.conf["ID"]}.npy',
                    np.asarray(avg_loss),
                )
            avg_ep_rews.append(
                np.mean(
                    [
                        np.sum(ep_rews)
                        for ep_rews in self.pr.logger["batch_rews"]
                    ]
                )
            )
            timesteps.append(self.pr.logger["t_so_far"])
            # Compute and save agreements
            with torch.no_grad():
                # The average distance from a single node to all of the other nodes in the problem
                actor_params = [torch.nn.utils.parameters_to_vector(self.pr.actors[i].parameters()) for i in range(self.pr.N)]
                critic_params = [torch.nn.utils.parameters_to_vector(self.pr.critics[i].parameters()) for i in range(self.pr.N)]

                # Stack all of the parameters into rows
                th_stack_a = torch.stack(actor_params)
                th_stack_c = torch.stack(critic_params)
                th_stack = torch.hstack((th_stack_a, th_stack_c))

                # Normalize the stack
                th_stack = torch.nn.functional.normalize(th_stack, dim=1)
                
                # Compute row-wise distances
                th_mean = torch.mean(th_stack, dim=0).reshape(1, -1)
                distances_mean = torch.cdist(th_stack, th_mean)
                agree_0 = np.append(agree_0, distances_mean[0].item())
                agree_1 = np.append(agree_1, distances_mean[1].item())
                agree_2 = np.append(agree_2, distances_mean[2].item())
            self.pr._log_summary()

            if profiler is not None:
                profiler.step()

            # Save our model if it's time
            if k % self.pr.save_freq == 0 or k == 5358:
                # marl
                # predator-prey
                torch.save(
                    {
                        "actor0": self.pr.actors[0].state_dict(),
                        "actor1": self.pr.actors[1].state_dict(),
                        "actor2": self.pr.actors[2].state_dict(),
                    },
                    f'./trained/ppo_actors_tag_kgt_{self.conf["ID"]}.pth',
                )
                torch.save(
                    {
                        "critic0": self.pr.critics[0].state_dict(),
                        "critic1": self.pr.critics[1].state_dict(),
                        "critic2": self.pr.critics[2].state_dict(),
                    },
                    f'./trained/ppo_critics_tag_kgt_{self.conf["ID"]}.pth',
                )

                # save plotting data
                np.save(
                    f'./trained/avg_ep_rews_kgt_{self.conf["ID"]}.npy',
                    np.asarray(avg_ep_rews),
                )
                np.save(
                    f'./trained/timesteps_kgt_{self.conf["ID"]}.npy',
                    np.asarray(timesteps),
                )
                np.savez(f'./trained/agreements_kgt_{self.conf["ID"]}', agree_0=agree_0, agree_1=agree_1, agree_2=agree_2)

            k += 1

        return
