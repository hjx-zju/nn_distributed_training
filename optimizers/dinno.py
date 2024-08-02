import torch
import math
from utils.quantize import quantize_
class DiNNO:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf

        self.duals = {
            i: torch.zeros((self.pr.n), device=device)
            for i in range(self.pr.N)
        }
        self.quant_bit=self.conf["quantize"]
        self.rho = self.conf["rho_init"]
        self.rho_scaling = self.conf["rho_scaling"]
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
    # def quantize(self,data,delta=1e-4,quantize=False):
    #     if quantize:
    #         return delta * torch.round(data /delta)
    #     else:
    #         return data
    def quantize(self,data,level=32,is_biased=False):
        if level!=32:
            s=2**level-1
            norm=data.norm(p=2)
            level_float=s*data.abs()/norm
            previous_level=torch.floor(level_float)
            is_next_level=(torch.rand_like(data)<(level_float-previous_level)).float()
            new_level=previous_level+is_next_level
            scale=1
            return scale * torch.sign(data) * norm * (new_level / s)
            
        else:
            return data
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
                    self.pr.models[i].parameters(), self.primal_lr[k],momentum=0.9
                )
            elif self.conf["primal_optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    self.pr.models[i].parameters(), self.primal_lr[k]
                )
            else:
                raise NameError("DiNNO primal optimizer is unknown.")

        for _ in range(self.pits):
            opt.zero_grad()

            # Model pass on the batch
            pred_loss = self.pr.local_batch_loss(i)

            # Get the primal variable WITH the autodiff graph attached.
            th = torch.nn.utils.parameters_to_vector(
                self.pr.models[i].parameters()
            )

            reg = torch.sum(
                torch.square(torch.cdist(th.reshape(1, -1), th_reg))
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            # if i==0:
            #     print("pred_loss: ",pred_loss.item()," dot_num: ", torch.dot(th, self.duals[i]).item(), " reg_num: ", self.rho * reg.item())
            loss.backward()
            opt.step()

        return

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics(at_end=(k == oits - 1))

            # Get the current primal variables
            ths = {
                i: torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            # Update the penalty parameter
            self.rho *= self.rho_scaling

            # Update the communication graph
            self.pr.update_graph()

            # Per node updates
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                thj = torch.stack([quantize_(ths[j],self.quant_bit) for j in neighs])

                self.duals[i] += self.rho * torch.sum(ths[i] - thj, dim=0)
                th_reg = (thj + ths[i]) / 2.0
                self.primal_update(i, th_reg, k)

            if profiler is not None:
                profiler.step()

        return
