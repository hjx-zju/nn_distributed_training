import torch
from utils import graph_generation
import copy
from utils.quantize import quantize_

class DSGT:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device
        self.quant_bit = self.conf["quantize"]
        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }

        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alpha = conf["alpha"]

        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.glists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
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
            W = W.to(self.device)
            bak_plist=copy.deepcopy(self.plists)
            # Iterate over the agents for communication step
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    # Update each parameter individually across all neighbors
                    for p in range(self.num_params):
                        # Ego update
                        self.plists[i][p].multiply_(W[i, i])
                        self.plists[i][p].add_(
                            self.ylists[i][p], alpha=-self.alpha * W[i, i]
                        )
                        # Neighbor updates
                        for j in neighs:
                            self.plists[i][p].add_(quantize_(bak_plist[j][p],self.quant_bit), alpha=W[i, j])
                            self.plists[i][p].add_(
                                quantize_(self.ylists[j][p],self.quant_bit), alpha=-self.alpha * W[i, j]
                            )
                self.pr.models[i].zero_grad()
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()
            bak_ylist=copy.deepcopy(self.ylists)
            bak_glist=copy.deepcopy(self.glists)
            # Compute the batch loss and update using the gradients
            for i in range(self.pr.N):
                # Batch loss
                # bloss = self.pr.local_batch_loss(i)
                # bloss.backward()

                neighs = list(self.pr.graph.neighbors(i))
                # print("node", i, " neighs: ", neighs)
                # print("node", i, " W: ", W[i, :])
                # Locally update model with gradient
                with torch.no_grad():
                    sum_ynorm = 0.0
                    sum_gnorm = 0.0
                    for p in range(self.num_params):
                     
                        self.ylists[i][p].multiply_(W[i, i])
                        for j in neighs:
                            self.ylists[i][p].add_(quantize_(bak_ylist[j][p],self.quant_bit), alpha=W[i, j])
                            self.ylists[i][p].add_(
                                quantize_(self.plists[j][p].grad,self.quant_bit), alpha=W[i, j]
                            )
                            self.ylists[i][p].add_(quantize_(bak_glist[j][p],self.quant_bit), alpha=-W[i, j])

                        sum_ynorm += torch.norm(self.ylists[i][p]).item()
                        self.ylists[i][p].add_(self.plists[i][p].grad,alpha=W[i,i])
                        self.ylists[i][p].add_(bak_glist[i][p], alpha=-W[i,i])
                        self.glists[i][p] = self.plists[i][p].grad.clone()

                        sum_gnorm += torch.norm(self.glists[i][p]).item()
            # for i in range(self.pr.N):
            #     with torch.no_grad():
            #         for p in range(self.num_params):
            #             self.plists[i][p].grad.zero_()
                #    print(
                #        "Node {} : ynorm {}, gnorm {}".format(
                #            i, sum_ynorm, sum_gnorm
                #        )
                #    )

            if profiler is not None:
                profiler.step()
        return
