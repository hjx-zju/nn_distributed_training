import torch
from utils import graph_generation
import copy
import random


class RANDCOM:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }

        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alpha = conf["alpha"]
        self.beta = conf["beta"]
        self.gamma = conf["gamma"]
        self.p = conf["p"]
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.glists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.zlists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.ylists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]

        # Initialize Ylists and Glists
        if self.conf["init_grads"]:
            for i in range(self.pr.N):
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()

                with torch.no_grad():
                    for p in range(self.num_params):
                        self.ylists[i][p] = self.plists[i][p].grad.detach().clone()
                        self.glists[i][p] = self.plists[i][p].grad.detach().clone()
                        self.plists[i][p].grad.zero_()

        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))

            # Compute graph weights
            W = graph_generation.get_metropolis(self.pr.graph)
            Wa = graph_generation.get_Wa(W, self.gamma)
            Wa = Wa.to(self.device)

            # Iterate over the agents for communication step
            for i in range(self.pr.N):
                self.pr.models[i].zero_grad()
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    # Update each parameter individually across all neighbors
                    for p in range(self.num_params):
                        # Ego update
                        self.zlists[i][p].set_(self.plists[i][p])
                        self.zlists[i][p].add_(
                            self.plists[i][p].grad, alpha=-self.alpha
                        )
                        self.zlists[i][p].add_(self.ylists[i][p], alpha=-1)

            # Compute the batch loss and update using the gradients
            flip = random.random()
            for i in range(self.pr.N):
                # flip coin
                if flip <= self.p:
                    neighs = list(self.pr.graph.neighbors(i))
                    with torch.no_grad():
                        for p in range(self.num_params):
                            self.plists[i][p].set_(Wa[i, i] * self.zlists[i][p])
                            # Neighbor updates
                            for j in neighs:
                                self.plists[i][p].add_(
                                    self.zlists[j][p], alpha=Wa[i, j]
                                )

                            self.ylists[i][p].add_(self.zlists[i][p], alpha=self.beta)
                            self.ylists[i][p].add_(self.plists[i][p], alpha=-self.beta)
                else:
                    with torch.no_grad():
                        for p in range(self.num_params):
                            self.plists[i][p].set_(self.zlists[i][p])

                # print("node", i, " neighs: ", neighs)
                # print("node", i, " W: ", W[i, :])
                # Locally update model with gradient

            if profiler is not None:
                profiler.step()
        return
