import torch
from utils import graph_generation
import copy


class FLEXGT:
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
        self.gamma = conf["gamma"]
   
        self.local_step=conf["t_local"]

        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.delta_grad = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
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
            for t in range(self.local_step):
                # Iterate over the agents for communication step
                for i in range(self.pr.N):
                    with torch.no_grad():
                        # Update each parameter individually across all neighbors
                        for p in range(self.num_params):
                            # Ego update
                            self.plists[i][p].add_(
                                self.ylists[i][p], alpha=-self.gamma
                            )
                            
                    self.pr.models[i].zero_grad()
                    bloss = self.pr.local_batch_loss(i)
                    bloss.backward()
                    with torch.no_grad():
                        for p in range(self.num_params):
                            self.ylists[i][p].add_(self.plists[i][p].grad)
                            self.ylists[i][p].add_(self.glists[i][p], alpha=-1.0)
                            self.glists[i][p] = self.plists[i][p].grad.clone().detach()
                  

                        
           
            bak_plists=copy.deepcopy(self.plists)
            bak_ylists=copy.deepcopy(self.ylists)

            for i in range(self.pr.N):
                
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    for p in range(self.num_params):
                        
                        self.plists[i][p].set_(W[i,i]*(bak_plists[i][p]-self.gamma*self.ylists[i][p]))
                        for j in neighs:
                            self.plists[i][p].add_(bak_plists[j][p]-self.gamma*self.ylists[j][p], alpha=W[i, j])
                self.pr.models[i].zero_grad()
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()
                with torch.no_grad():
                    for p in range(self.num_params):
                        self.ylists[i][p].multiply_(W[i,i])
                        for j in neighs:
                            self.ylists[i][p].add_(bak_ylists[j][p], alpha=W[i, j])
                        self.ylists[i][p].add_(self.plists[i][p].grad)
                        self.ylists[i][p].add_(self.glists[i][p], alpha=-1.0)
                        self.glists[i][p] = self.plists[i][p].grad.clone().detach()
                

                       

                        
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

    # def train(self, profiler=None):
    #     eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
    #     oits = self.conf["outer_iterations"]

    #     # Initialize Ylists and Glists
    #     if self.conf["init_grads"]:
    #         for i in range(self.pr.N):
    #             bloss = self.pr.local_batch_loss(i)
    #             bloss.backward()

    #             with torch.no_grad():
    #                 for p in range(self.num_params):
    #                     self.ylists[i][p] = (
    #                         self.alpha * self.plists[i][p].grad.detach().clone()
    #                     )
    #                     self.glists[i][p] = self.plists[i][p].grad.detach().clone()
    #                     self.plists[i][p].grad.zero_()

    #     # Optimization loop
    #     for k in range(oits):
    #         if k % eval_every == 0 or k == oits - 1:
    #             self.pr.evaluate_metrics(at_end=(k == oits - 1))

    #         # Compute graph weights
    #         W = graph_generation.get_metropolis(self.pr.graph)
    #         W = W.to(self.device)
    #         if k % 3== 0:
    #             bak_plist = copy.deepcopy(self.plists)

    #             for i in range(self.pr.N):
    #                 neighs = list(self.pr.graph.neighbors(i))
    #                 with torch.no_grad():

    #                     for p in range(self.num_params):
    #                         self.plists[i][p].multiply_(W[i, i])
    #                         self.plists[i][p].add_(
    #                             self.ylists[i][p], alpha=-self.gamma * W[i, i]
    #                         )
    #                         # Neighbor updates
    #                         for j in neighs:
    #                             self.plists[i][p].add_(bak_plist[j][p], alpha=W[i, j])
    #                             self.plists[i][p].add_(
    #                                 self.ylists[j][p], alpha=-self.gamma * W[i, j]
    #                             )
    #                 self.pr.models[i].zero_grad()
    #                 bloss = self.pr.local_batch_loss(i)
    #                 bloss.backward()
    #             bak_ylist = copy.deepcopy(self.ylists)
    #             bak_glist = copy.deepcopy(self.glists)
    #             for i in range(self.pr.N):
    #                 neighs = list(self.pr.graph.neighbors(i))

    #                 with torch.no_grad():
    #                     for p in range(self.num_params):

    #                         self.ylists[i][p].multiply_(W[i, i])
    #                         for j in neighs:
    #                             self.ylists[i][p].add_(bak_ylist[j][p], alpha=W[i, j])
    #                             # self.ylists[i][p].add_(
    #                             #     self.plists[j][p].grad, alpha=self.alpha * W[i, j]
    #                             # )
    #                             # self.ylists[i][p].add_(
    #                             #     bak_glist[j][p], alpha=-self.alpha * W[i, j]
    #                             # )

    #                         self.ylists[i][p].add_(
    #                             self.plists[i][p].grad
    #                         )
    #                         self.ylists[i][p].add_(
    #                             bak_glist[i][p],alpha=-1
    #                         )
    #                         self.glists[i][p] = self.plists[i][p].grad.clone().detach()
    #                         # self.plists[i][p].grad.zero_()
    #         else:
    #             # print("local update")
    #             for i in range(self.pr.N):
    #                 # # Batch loss
    #                 # if l==0 and k!=0:
    #                 #     bloss = self.pr.local_batch_loss(i)
    #                 #     bloss.backward()
    #                 #     with torch.no_grad():
    #                 #         for p in range(self.num_params):
    #                 #             self.glists[i][p]=self.plists[i][p].grad.clone().detach()
    #                 #             self.plists[i][p].grad.zero_()
    #                 with torch.no_grad():
    #                     for p in range(self.num_params):
    #                         self.plists[i][p].add_(self.ylists[i][p], alpha=-self.gamma)
    #                         # self.plists[i][p].grad.zero_()
    #                 self.pr.models[i].zero_grad()
    #                 bloss = self.pr.local_batch_loss(i)
    #                 bloss.backward()
    #                 with torch.no_grad():
    #                     for p in range(self.num_params):
    #                         # self.delta_grad[i][p]=self.plists[i][p].grad.clone()-self.glists[i][p].clone()
    #                         self.ylists[i][p].add(
    #                             self.plists[i][p].grad, alpha=self.alpha
    #                         )
    #                         self.ylists[i][p].add_(self.glists[i][p], alpha=-self.alpha)
    #                         self.glists[i][p] = self.plists[i][p].grad.clone().detach()
    #                         self.plists[i][p].grad.zero_()

    #                 # self.ylists[i][p].add_(self.plists[i][p].grad)
    #                 # self.ylists[i][p].add_(self.glists[i][p], alpha=-1.0)
    #                 # self.glists[i][p] = self.plists[i][p].grad.clone()
    #                 # Compute the batch loss and update using the gradients

    #             # with torch.no_grad():
    #             #     for p in range(self.num_params):
    #             #         self.plists[i][p].grad.zero_()

    #         if profiler is not None:
    #             profiler.step()
    #     return

