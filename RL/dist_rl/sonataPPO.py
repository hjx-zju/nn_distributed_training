import torch
import math
import numpy as np
import copy
from utils import graph_generation


class SONATAPPO:
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
        self.tau = conf["tau"]
        self.alpha = conf["alpha"]
        # self.critic_lr=conf["primal_lr_finish"]
        if self.conf["lr_decay_type"] == "constant":
            self.primal_lr = self.conf["primal_lr_start"] * torch.ones(
                self.conf["outer_iterations"]
            )
        elif self.conf["lr_decay_type"] == "linear":
            self.primal_lr = torch.linspace(
                self.conf["primal_lr_start"],
                self.conf["primal_lr_finish"],
                self.conf["outer_iterations"],
            )
        elif self.conf["lr_decay_type"] == "log":
            self.primal_lr = torch.logspace(
                math.log(self.conf["primal_lr_start"], 10),
                math.log(self.conf["primal_lr_finish"], 10),
                self.conf["outer_iterations"],
            )
        else:
            raise NameError("Unknow primal learning rate decay type.")
        if conf["critic_lr_decay_type"] == "constant":
            self.critic_lr = conf["critic_lr_start"] * torch.ones(
                conf["outer_iterations"]
            )
        elif conf["critic_lr_decay_type"] == "linear":
            self.critic_lr = torch.linspace(
                conf["critic_lr_start"],
                conf["critic_lr_finish"],
                conf["outer_iterations"],
            )
        elif conf["critic_lr_decay_type"] == "log":
            self.critic_lr = torch.logspace(
                math.log(conf["critic_lr_start"], 10),
                math.log(conf["critic_lr_finish"], 10),
                conf["outer_iterations"],
            )
        self.pits = self.conf["primal_iterations"]

        if self.conf["persistant_primal_opt"]:
            self.opts = {}
            for i in range(self.pr.N):
                if self.conf["primal_optimizer"] == "adam":
                    self.opts[i] = torch.optim.Adam(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    self.opts[i] = torch.optim.SGD(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    self.opts[i] = torch.optim.AdamW(
                        self.pr.models[i].parameters(), self.primal_lr[0]
                    )
                else:
                    raise NameError("CADMM primal optimizer is unknown.")

    def primal_update(self, i, k):
        # if self.conf["persistant_primal_opt"]:
        #    opt = self.opts[i]
        # else:
        #    if self.conf["primal_optimizer"] == "adam":
        #        opt = torch.optim.Adam(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    elif self.conf["primal_optimizer"] == "sgd":
        #        opt = torch.optim.SGD(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    elif self.conf["primal_optimizer"] == "adamw":
        #        opt = torch.optim.AdamW(
        #            self.pr.models[i].parameters(), self.primal_lr[k]
        #        )
        #    else:
        #        raise NameError("CADMM primal optimizer is unknown.")

        opt_actor = torch.optim.Adam(
            self.pr.actors[i].parameters(), lr=self.primal_lr[k]
        )
        opt_critic = torch.optim.Adam(
            self.pr.critics[i].parameters(), lr=self.critic_lr[k]
        )
        ori_th_actor = (
            torch.nn.utils.parameters_to_vector(self.pr.actors[i].parameters())
            .clone()
            .detach()
        )
        delta_grad_actor = torch.nn.utils.parameters_to_vector(
            self.ylists_actor[i]
        ) - torch.nn.utils.parameters_to_vector(self.glists_actor[i])
        ori_th_critic = (
            torch.nn.utils.parameters_to_vector(self.pr.critics[i].parameters())
            .clone()
            .detach()
        )
        delta_grad_critic = torch.nn.utils.parameters_to_vector(
            self.ylists_critic[i]
        ) - torch.nn.utils.parameters_to_vector(self.glists_critic[i])

        for _ in range(self.pits):

            # Model pass on the batch
            # pred_loss = self.pr.local_batch_loss(i)
            actor_loss, critic_loss = self.pr.ev_ppo_loss(i)

            # Get the primal variable WITH the autodiff graph attached.
            th_actor = torch.nn.utils.parameters_to_vector(
                self.pr.actors[i].parameters()
            )

            th_critic = torch.nn.utils.parameters_to_vector(
                self.pr.critics[i].parameters()
            )
            surrogate_loss_actor = (
                self.tau / 2 * torch.square(torch.norm(th_actor - ori_th_actor))
            )
            surrogate_loss_critic = (
                self.tau / 2 * torch.square(torch.norm(th_critic - ori_th_critic))
            )
            dot_actor = torch.dot(delta_grad_actor, th_actor)
            dot_critic = torch.dot(delta_grad_critic, th_critic)

            aloss = actor_loss + surrogate_loss_actor + dot_actor
            closs = critic_loss + surrogate_loss_critic + dot_critic
            # if i == 0:
            #     print(
            #         f"\nActor loss: {actor_loss.item()} Surrogate loss: {surrogate_loss_actor.item()} Dot product: {dot_actor.item()}"
            #     )
            #     print(
            #         f"Critic loss: {critic_loss.item() } Surrogate loss: {surrogate_loss_critic.item()} Dot product: {dot_critic.item()}"
            #     )
            opt_actor.zero_grad()
            aloss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.pr.actors[i].parameters(), 0.5)
            opt_actor.step()

            opt_critic.zero_grad()
            closs.backward()
            torch.nn.utils.clip_grad_norm_(self.pr.critics[i].parameters(), 0.5)
            opt_critic.step()

        return

    def train(self, profiler=None):
        # eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]

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
                        self.plists_actor[i][p].grad.detach().clone()
                    )
                    self.glists_actor[i][p] = (
                        self.plists_actor[i][p].grad.detach().clone()
                    )
                    self.plists_actor[i][p].grad.zero_()

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pr.critics[i].parameters(), 0.5)
            with torch.no_grad():
                for p in range(self.num_params_critic):
                    self.ylists_critic[i][p] = (
                        self.plists_critic[i][p].grad.detach().clone()
                    )
                    self.glists_critic[i][p] = (
                        self.plists_critic[i][p].grad.detach().clone()
                    )
                    self.plists_critic[i][p].grad.zero_()
        k = 0
        avg_ep_rews = []
        avg_loss = []
        timesteps = []
        agree_0 = np.array([])
        agree_1 = np.array([])
        agree_2 = np.array([])
        while self.pr.logger["t_so_far"] < self.conf["max_rl_timesteps"]:
            self.pr.split_rollout_marl()
            self.pr.update_advantage()

            # Per node updates
            for i in range(self.pr.N):
                self.primal_update(i, k)

            ths_actor = {
                i: torch.nn.utils.parameters_to_vector(self.pr.actors[i].parameters())
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            ths_critic = {
                i: torch.nn.utils.parameters_to_vector(self.pr.critics[i].parameters())
                .clone()
                .detach()
                for i in range(self.pr.N)
            }
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    sum_actor = W[i, i] * ths_actor[i]
                    sum_critic = W[i, i] * ths_critic[i]
                    for j in neighs:
                        sum_actor += W[i, j] * ths_actor[j]
                        sum_critic += W[i, j] * ths_critic[j]
                    torch.nn.utils.vector_to_parameters(
                        sum_actor, self.pr.actors[i].parameters()
                    )
                    torch.nn.utils.vector_to_parameters(
                        sum_critic, self.pr.critics[i].parameters()
                    )

            bak_ylist_actor = copy.deepcopy(self.ylists_actor)
            bak_ylist_critic = copy.deepcopy(self.ylists_critic)

            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                actor_loss, critic_loss = self.pr.ev_ppo_loss(i)
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.pr.actors[i].parameters(), 0.5)
                
                with torch.no_grad():
                    for p in range(self.num_params_actor):
                        self.ylists_actor[i][p].multiply_(W[i, i])
                        self.ylists_actor[i][p].add_(self.plists_actor[i][p].grad)
                        self.ylists_actor[i][p].add_(
                            self.glists_actor[i][p], alpha=-1.0
                        )
                        for j in neighs:
                            self.ylists_actor[i][p].add_(
                                bak_ylist_actor[j][p], alpha=W[i, j]
                            )
                        self.glists_actor[i][p] = self.plists_actor[i][p].grad.clone().detach()
                        self.plists_actor[i][p].grad.zero_()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pr.critics[i].parameters(), 0.5)
                with torch.no_grad():
                    for p in range(self.num_params_critic):
                        self.ylists_critic[i][p].multiply_(W[i, i])
                        self.ylists_critic[i][p].add_(self.plists_critic[i][p].grad)
                        self.ylists_critic[i][p].add_(
                            self.glists_critic[i][p], alpha=-1.0
                        )
                        for j in neighs:
                            self.ylists_critic[i][p].add_(
                                bak_ylist_critic[j][p], alpha=W[i, j]
                            )
                        self.glists_critic[i][p] = self.plists_critic[i][p].grad.clone().detach()
                        self.plists_critic[i][p].grad.zero_()
            # for i in range(self.pr.N):
            #     with torch.no_grad():
            #         for p in range(self.num_params_actor):
            #             self.glists_actor[i][p] = (
            #                 self.plists_actor[i][p].grad.clone().detach()
            #             )
            #         for p in range(self.num_params_critic):
            #             self.glists_critic[i][p] = (
            #                 self.plists_critic[i][p].grad.clone().detach()
            #             )

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
            # if k % 10 == 0:
            #     np.save(
            #         f'./results_rl/avg_loss_sonata{self.conf["ID"]}.npy',
            #         np.asarray(avg_loss),
            #     )

            avg_ep_rews.append(
                np.mean([np.sum(ep_rews) for ep_rews in self.pr.logger["batch_rews"]])
            )
            timesteps.append(self.pr.logger["t_so_far"])
            # Compute and save agreements
            with torch.no_grad():
                # The average distance from a single node to all of the other nodes in the problem
                actor_params = [
                    torch.nn.utils.parameters_to_vector(self.pr.actors[i].parameters())
                    for i in range(self.pr.N)
                ]
                critic_params = [
                    torch.nn.utils.parameters_to_vector(self.pr.critics[i].parameters())
                    for i in range(self.pr.N)
                ]

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
                    f'./results_sonata/ppo_actors_tag_sonata_{self.conf["ID"]}_{k}.pth',
                )
                torch.save(
                    {
                        "critic0": self.pr.critics[0].state_dict(),
                        "critic1": self.pr.critics[1].state_dict(),
                        "critic2": self.pr.critics[2].state_dict(),
                    },
                    f'./results_sonata/ppo_critics_tag_sonata_{self.conf["ID"]}_{k}.pth',
                )

                # save plotting data
                np.save(
                    f'./results_sonata/avg_ep_rews_sonata_{self.conf["ID"]}.npy',
                    np.asarray(avg_ep_rews),
                )
                np.save(
                    f'./results_sonata/timesteps_sonata_{self.conf["ID"]}.npy',
                    np.asarray(timesteps),
                )
                np.savez(
                    f'./results_sonata/agreements_sonata_{self.conf["ID"]}',
                    agree_0=agree_0,
                    agree_1=agree_1,
                    agree_2=agree_2,
                )

            k += 1

        return
