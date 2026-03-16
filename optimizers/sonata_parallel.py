import torch
import torch.backends
import torch.backends.cudnn
from utils import graph_generation
import copy
import math
from utils.quantize import quantize_
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils.graph_manager import My_Graph
class SONATA_PARALLEL:

    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device
        
                

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
    
    def train(self):
        torch.backends.cudnn.deterministic = True
        mp.spawn(worker, nprocs=self.pr.N,args=(self.pr,self.conf,self.device))

def local_update(i,primal_lr,parameter,parameter_2,y_estimator,g_estimator,pr,conf):
    if conf["primal_optimizer"] == "adam":
        opt = torch.optim.Adam(
           parameter, primal_lr
        )
    elif conf["primal_optimizer"] == "sgd":
        opt = torch.optim.SGD(
           parameter, primal_lr
        )
    elif conf["primal_optimizer"] == "adamw":
        opt = torch.optim.AdamW(
           parameter, primal_lr
        )
    else:
        raise NameError(" primal optimizer is unknown.")
    ori_theta=torch.nn.utils.parameters_to_vector(parameter).clone().detach()
    delta_grad=torch.nn.utils.parameters_to_vector(y_estimator)-torch.nn.utils.parameters_to_vector(g_estimator)
    pits = conf["primal_iterations"]
    for _ in range(pits):
        opt.zero_grad()
        pred_loss = pr.local_batch_loss(i)
        theta=torch.nn.utils.parameters_to_vector(parameter)
        surrogate_loss=conf["tau"]/2*torch.square(torch.norm(theta-ori_theta))
        loss=pred_loss+surrogate_loss+torch.dot(delta_grad,theta)
        loss.backward()
        opt.step()
    opt.zero_grad()
    alpha=conf["alpha"]
    parameter_2=(1-alpha)*ori_theta+alpha*torch.nn.utils.parameters_to_vector(parameter).detach().clone()
    return    
def worker(i,prob_conf,opt_conf,device,train_subsets,val_set):
    
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:33069", rank=i, world_size=pr.N)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            num_gpu = torch.cuda.device_count()
            torch.cuda.set_device(i % num_gpu)
        parameter=pr.models[i].parameters()
        parameter_2=torch.nn.utils.parameters_to_vector(parameter).clone().detach()
        num_params = len(parameter)
        alpha=conf["alpha"]
        tau=conf["tau"]
        use_prox=conf["use_prox"]
        quant_bit=conf["quantize"]
        base_zeros = torch.zeros_like(parameter, requires_grad=False, device=device)
        g_estimator=copy.deepcopy(base_zeros)
        y_estimator=copy.deepcopy(base_zeros)

        eval_every = pr.conf["metrics_config"]["evaluate_frequency"]
        oits = conf["outer_iterations"]
        if conf["lr_decay_type"] == "constant":
            primal_lr = conf["primal_lr_start"] * torch.ones(
                conf["outer_iterations"]
            )
        elif conf["lr_decay_type"] == "linear": 
            primal_lr = torch.linspace(
                conf["primal_lr_start"],
                conf["primal_lr_finish"],
                conf["outer_iterations"],
            )
        elif conf["lr_decay_type"] == "log":
            primal_lr = torch.logspace(
                math.log(conf["primal_lr_start"], 10),
                math.log(conf["primal_lr_finish"], 10),
                conf["outer_iterations"],
            )
        else:
            raise NameError("Unknow primal learning rate decay type.")
        pits = conf["primal_iterations"]

        # Initialize Ylists and Glists
        if conf["init_grads"]:
            bloss = pr.local_batch_loss(i)
            bloss.backward()

            with torch.no_grad():
                for p in range(num_params):
                    y_estimator[p] = (
                        parameter[p].grad.detach().clone()
                    )
                    g_estimator[p] = (
                        parameter[p].grad.detach().clone()
                    )
                    parameter[p].grad.zero_()
        W = graph_generation.get_metropolis(pr.graph)
        graph=My_Graph(rank=i,world_size=pr.N,weight_matrix=W)
        out_edges, in_edges = graph.get_edges()
        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                if i==0:
                    pr.evaluate_metrics(at_end=(k == oits - 1))
                print("rank: ",i," iteration: ",k)
                
            pr.update_graph()

            # Compute graph weights
            W = graph_generation.get_metropolis(pr.graph)
            W = W.to(device)
            
            local_update(i,primal_lr[k],parameter,parameter_2,y_estimator,g_estimator,pr,conf,device)
                
            send_msg=parameter_2.clone().detach()
            for out_edge in out_edges:
                assert i==out_edge.src
                weight=W[out_edge.dest,i].to(i)
                dist.broadcast(send_msg.mul(weight),src=i,group=out_edge.process_group,async_op=True)
            in_msg=W[i,i]*parameter_2
            recv_msg=torch.zeros_like(parameter_2)
            for in_edge in in_edges:
                dist.broadcast(tensor=recv_msg,src=in_edge.src,group=in_edge.process_group)
                in_msg.add_(recv_msg)
            # Iterate over the agents for communication step
            torch.nn.utils.vector_to_parameters(in_msg,parameter)
   
        
            bloss = pr.local_batch_loss(i)
            bloss.backward()
            
            bak_ylist=copy.deepcopy(y_estimator)
            bak_glist=copy.deepcopy(g_estimator)
            # Compute the batch loss and update using the gradients
            y_send_msg=torch.nn.utils.parameters_to_vector(y_estimator).clone().detach()
            for out_edge in out_edges:
                assert i==out_edge.src
                weight=W[out_edge.dest,i].to(i)
                dist.broadcast(y_send_msg.mul(weight),src=i,group=out_edge.process_group,async_op=True)
            
            in_y_msg=W[i,i]*torch.nn.utils.parameters_to_vector(y_estimator)
            recv_y_msg=torch.zeros_like(y_send_msg)
            for in_edge in in_edges:
                dist.broadcast(tensor=recv_y_msg,src=in_edge.src,group=in_edge.process_group)
                in_y_msg.add_(recv_y_msg)
            torch.nn.utils.vector_to_parameters(in_y_msg,y_estimator)
            for p in range(num_params):
                y_estimator[p].add_(parameter[p].grad)
                y_estimator[p].add_(g_estimator[p], alpha=-1.0)
                g_estimator[p] = parameter[p].grad.clone().detach()


        return       
    
