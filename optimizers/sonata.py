import torch
from utils import graph_generation
import copy
import math
from utils.quantize import quantize_

class SONATA:

    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device
        
        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }
        self.x_2_lists={i: torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach() for i in range(self.pr.N)}
        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alpha = conf["alpha"]
        self.tau=conf["tau"]
        self.use_prox=conf["use_prox"]
        self.quant_bit=self.conf["quantize"]
        
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.glists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.ylists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
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
                

    def local_update(self,i,k):
        if self.use_prox:
            ori_theta=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach()
            grad=torch.nn.utils.parameters_to_vector(self.ylists[i])
            self.x_2_lists[i]=ori_theta-1/self.tau*grad
        else:
            if self.conf["persistant_primal_opt"]:
                opt = self.opts[i]
            else:
                if self.conf["primal_optimizer"] == "adam":
                    opt = torch.optim.Adam(
                        self.pr.models[i].parameters(), self.primal_lr[k]
                    )
                elif self.conf["primal_optimizer"] == "sgd":
                    opt = torch.optim.SGD(
                        self.pr.models[i].parameters(), self.primal_lr[k]
                    )
                elif self.conf["primal_optimizer"] == "adamw":
                    opt = torch.optim.AdamW(
                        self.pr.models[i].parameters(), self.primal_lr[k]
                    )
                else:
                    raise NameError("CADMM primal optimizer is unknown.")
            ori_theta=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach()
            delta_grad=torch.nn.utils.parameters_to_vector(self.ylists[i])-torch.nn.utils.parameters_to_vector(self.glists[i])
            
            # print(delta_grad[:10])
            for _ in range(self.pits):
                opt.zero_grad()
                pred_loss = self.pr.local_batch_loss(i)
                theta=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters())
                surrogate_loss=self.tau/2*torch.square(torch.norm(theta-ori_theta))
                loss=pred_loss+surrogate_loss+torch.dot(delta_grad,theta)
                # if i==0:
                #     print("predict loss: ",pred_loss.item()," surrogate loss: ",surrogate_loss.item()," delta_grad: ",torch.dot(delta_grad,theta).item())
                loss.backward()
                opt.step()
            opt.zero_grad()
            self.x_2_lists[i]=(1-self.alpha)*ori_theta+self.alpha*torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).detach().clone()
        return
    
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
                        self.ylists[i][p] = (
                            self.plists[i][p].grad.detach().clone()
                        )
                        self.glists[i][p] = (
                            self.plists[i][p].grad.detach().clone()
                        )
                        self.plists[i][p].grad.zero_()

        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))
                
            self.pr.update_graph()

            # Compute graph weights
            W = graph_generation.get_metropolis(self.pr.graph)
            W = W.to(self.device)
            
            for i in range(self.pr.N):
                self.local_update(i,k)
                
            # Iterate over the agents for communication step
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    # Update each parameter individually across all neighbors
                    sum=W[i,i]*self.x_2_lists[i]
                    # Neighbor updates
                    for j in neighs:
                        sum+=W[i,j]*quantize_(self.x_2_lists[j],self.quant_bit)
                        
                    # origin=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach()
                    torch.nn.utils.vector_to_parameters(sum,self.pr.models[i].parameters()) 
                    # new=torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach()
                    # if(torch.equal(origin,new)):
                    #    raise NameError("node ",i," not updated")
                        
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()
            bak_ylist=copy.deepcopy(self.ylists)
            bak_glist=copy.deepcopy(self.glists)
            # Compute the batch loss and update using the gradients
            for i in range(self.pr.N):
                # Batch loss

                neighs = list(self.pr.graph.neighbors(i))

                with torch.no_grad():
                    sum_ynorm = 0.0
                    sum_gnorm = 0.0
                    for p in range(self.num_params):
                        self.ylists[i][p].multiply_(W[i, i])
          
                        self.ylists[i][p].add_(self.plists[i][p].grad)
                        self.ylists[i][p].add_(self.glists[i][p], alpha=-1.0)
                        
                        for j in neighs:
                            self.ylists[i][p].add_(
                                quantize_(bak_ylist[j][p],self.quant_bit), alpha=W[i, j]
                            )
                            # self.ylists[i][p].add_(quantize_(self.plists[j][p].grad,self.quant_bit), alpha=W[i, j])
                            # self.ylists[i][p].add_(quantize_(self.glists[j][p],self.quant_bit), alpha=-W[i, j])

                        # self.glists[i][p] = self.plists[i][p].grad.clone()
            #             self.plists[i][p].grad.zero_()

                        sum_ynorm += torch.norm(self.ylists[i][p]).item()


                        sum_gnorm += torch.norm(self.glists[i][p]).item()
                    # if i==0:
                    #     print(
                    #         "Node {} : ynorm {}, gnorm {}".format(
                    #             i, sum_ynorm, sum_gnorm
                    #         )
                    #     )
                    
            for i in range(self.pr.N):
                with torch.no_grad():
                    for p in range(self.num_params):
                        self.glists[i][p] = self.plists[i][p].grad.clone().detach()
            #             self.plists[i][p].grad.zero_()

            if profiler is not None:
                profiler.step()
        return
