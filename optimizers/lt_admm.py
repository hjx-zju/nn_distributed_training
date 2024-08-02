import torch

import math
import copy

def gradients_to_vector(parameters):
    vec = []
    for param in parameters:
        if param.grad is not None:
            vec.append(param.grad.view(-1))
    return torch.cat(vec)
class LT_ADMM:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf

        self.duals = {
            i: torch.zeros((self.pr.n), device=device)
            for i in range(self.pr.N)
        }
        self.z={i:{j: torch.nn.utils.parameters_to_vector(self.pr.models[i].parameters()).clone().detach() for j in list(self.pr.graph.neighbors(i))} for i in range(self.pr.N)}
        self.bak_z=copy.deepcopy(self.z)
        self.rho = self.conf["rho_init"]
        self.rho_scaling = self.conf["rho_scaling"]
        self.use_opt=self.conf["use_opt"]
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

    def primal_update(self, i, th_reg, k):
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
                raise NameError("DiNNO primal optimizer is unknown.")
        if self.use_opt:
            for _ in range(self.pits):
                
                    opt.zero_grad()

                    # Model pass on the batch
                    pred_loss = self.pr.local_batch_loss(i)

                    # Get the primal variable WITH the autodiff graph attached.
                    th = torch.nn.utils.parameters_to_vector(
                        self.pr.models[i].parameters()
                    )

                    reg = torch.square(torch.cdist(th.reshape(1, -1), th_reg.reshape(1,-1)))
                    

                    loss = pred_loss  + 1/2*len(list(self.pr.graph.neighbors(i)))*self.rho * reg
                    loss.backward()
                    opt.step()
        else:
            for _ in range(self.pits):
                self.pr.models[i].zero_grad()
              
                pred_loss = self.pr.local_batch_loss(i)
                pred_loss.backward()
                th = torch.nn.utils.parameters_to_vector(
                            self.pr.models[i].parameters()
                        )
                grad=gradients_to_vector(self.pr.models[i].parameters())
                update_th=th-0.01*(grad+self.rho*len(list(self.pr.graph.neighbors(i)))*(th-th_reg))
                torch.nn.utils.vector_to_parameters(update_th, self.pr.models[i].parameters())
                

        return

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))


            # Update the penalty parameter
            # self.rho *= self.rho_scaling

            # Update the communication graph
            self.pr.update_graph()

            # Per node updates
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                thj = sum(self.z[i].values())/self.rho/len(neighs)
                
                self.primal_update(i, thj, k)
            for i in range(self.pr.N):
                for j in list(self.pr.graph.neighbors(i)):
                    with torch.no_grad():
                        self.z[i][j]=1/2*self.bak_z[i][j]-1/2*(self.bak_z[j][i]-2*self.rho*torch.nn.utils.parameters_to_vector(self.pr.models[j].parameters()).clone().detach())
            
            self.bak_z=copy.deepcopy(self.z)
            if profiler is not None:
                profiler.step()

        return
