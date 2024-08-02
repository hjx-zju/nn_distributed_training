import torch
from utils import graph_generation
import copy
import math
from utils.quantize import quantize_
class KGT:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }
        self.quant_bit=self.conf["quantize"]
        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alpha = conf["alpha"]
        self.gamma=conf["gamma"]
        self.local_step=conf["t_local"]
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.glists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.ylists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.clists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        
        self.lr=torch.logspace(math.log(conf["lr_start"],10),math.log(conf["lr_end"],10),conf["outer_iterations"])
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
                        self.ylists[i][p] = -self.plists[i][p].grad.detach().clone()

            for i in range(self.pr.N):
                with torch.no_grad():
                    neighs = list(self.pr.graph.neighbors(i))
                    num_neighs = len(neighs)+1
                    for p in range(self.num_params):
                        self.ylists[i][p]+=self.plists[i][p].grad.detach().clone()/num_neighs
                        for j in neighs:
                            self.ylists[i][p]+=self.plists[j][p].grad.detach().clone()/num_neighs
        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))

            # Compute graph weights
            W = graph_generation.get_metropolis(self.pr.graph)
            W = W.to(self.device)
            bak_plists=copy.deepcopy(self.plists)
            use_adam=False
            for i in range(self.pr.N):
                opt=torch.optim.Adam(self.pr.models[i].parameters(),lr=self.lr[k])
                for t in range(self.local_step):
                    if use_adam:
                        opt.zero_grad()
                        bloss=self.pr.local_batch_loss(i)
                        th=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters())
                        c=torch.nn.utils.parameters_to_vector(self.clists[i]).clone().detach()
                        loss=bloss+torch.dot(c,th)
                        loss.backward()
                        opt.step()
                # Iterate over the agents for communication step
                    else:
                        self.pr.models[i].zero_grad()
                        bloss = self.pr.local_batch_loss(i)
                        bloss.backward()
                        with torch.no_grad():
                            # Update each parameter individually across all neighbors
                            for p in range(self.num_params):
                                # Ego update
                                self.plists[i][p].add_(
                                    self.plists[i][p].grad, alpha=-self.gamma
                                )
                                self.plists[i][p].add_(self.clists[i][p],alpha=-self.gamma)
                  
            for i in range(self.pr.N):
                with torch.no_grad():
                    for p in range(self.num_params):
                        self.ylists[i][p].zero_()
                        self.ylists[i][p].add_(bak_plists[i][p],alpha=1/self.local_step/self.gamma)
                        self.ylists[i][p].add_(self.plists[i][p],alpha=-1/self.local_step/self.gamma)
                        
            # Compute the batch loss and update using the gradients
            # bak_ylist=copy.deepcopy(self.ylists)

            for i in range(self.pr.N):

                neighs = list(self.pr.graph.neighbors(i))
                # print("node", i, " neighs: ", neighs)
                # print("node", i, " W: ", W[i, :])
                # Locally update model with gradient
                with torch.no_grad():
                    
                    for p in range(self.num_params):
                        self.clists[i][p].add_(self.ylists[i][p],alpha=W[i,i]-1)
                        self.plists[i][p].set_(W[i,i]*(bak_plists[i][p]-self.local_step*self.gamma*self.alpha*self.ylists[i][p]))
                        for j in neighs:
                            self.clists[i][p].add_(quantize_(self.ylists[j][p],self.quant_bit), alpha=W[i, j])
                            self.plists[i][p].add_(quantize_(bak_plists[j][p]-self.local_step*self.gamma*self.alpha*self.ylists[j][p],self.quant_bit), alpha=W[i, j])
                       # self.plists[i][p].add_(-alph * self.plists[i][p].grad

                       

                        
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
