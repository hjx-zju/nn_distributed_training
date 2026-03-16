import os
import sys
from datetime import datetime
from shutil import copyfile
import copy
import math
import yaml
from torchvision import datasets, transforms
import torch
import networkx as nx
sys.path.append('../')
from models.mnist_conv_nn import MNISTConvNet
from problems.dist_mnist_problem import DistMNISTProblem,DistMNISTProblem_Single
from optimizers.dinno import DiNNO
from optimizers.dsgd import DSGD
from optimizers.dsgt import DSGT
from optimizers.sonata import SONATA 

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils.graph_manager import My_Graph
from utils import graph_generation

torch.set_default_tensor_type(torch.DoubleTensor)



def experiment(yaml_pth):
    # load the config yaml
    with open(yaml_pth) as f:
        conf_dict = yaml.safe_load(f)

    # Seperate configuration groups
    exp_conf = conf_dict["experiment"]

    # Create the output directory
    output_metadir = exp_conf["output_metadir"]
    if not os.path.exists(output_metadir):
        os.mkdir(output_metadir)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(
        output_metadir, time_now + "_" + exp_conf["name"]
    )
    seed=exp_conf["seed"]
    

    if exp_conf["writeout"]:
        os.mkdir(output_dir)
        copyfile(yaml_pth, os.path.join(output_dir, time_now + ".yaml"))
    exp_conf["output_dir"] = output_dir  # probably bad practice

    # Create communication graph
    graph_conf = exp_conf["graph"]
    N, graph = graph_generation.generate_from_conf(graph_conf)
    # N=10
    if exp_conf["writeout"]:
        # Save the graph for future visualization
        nx.write_gpickle(graph, os.path.join(output_dir, "graph.gpickle"))

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_dir = exp_conf["data_dir"]
    joint_train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_set = datasets.MNIST(data_dir, train=False, transform=transform)

    if exp_conf["data_split_type"] == "random":
        num_samples_per = len(joint_train_set.targets) / N
        joint_splits = [int(num_samples_per) for _ in range(N)]
        train_subsets = torch.utils.data.random_split(
            joint_train_set, joint_splits
        )
    elif exp_conf["data_split_type"] == "hetero":
        classes = torch.unique(joint_train_set.targets)
        train_subsets = []
        if N <= len(classes):
            joint_labels = joint_train_set.targets
            node_classes = torch.split(classes, int(len(classes) / N))
            for i in range(N):
                # from here: https://discuss.pytorch.org/t/tensor-indexing-with-conditions/81297/2
                locs = [lab == joint_labels for lab in node_classes[i]]
                idx_keep = torch.nonzero(torch.stack(locs).sum(0)).reshape(-1)
                train_subsets.append(
                    torch.utils.data.Subset(joint_train_set, idx_keep)
                )
        else:
            raise NameError("Hetero MNIST N > 10 not supported.")

    # Create base model
    model_conf = exp_conf["model"]
    base_model = MNISTConvNet(
        model_conf["num_filters"],
        model_conf["kernel_size"],
        model_conf["linear_width"],
    )

    # Define base loss function
    if exp_conf["loss"] == "NLL":
        base_loss = torch.nn.NLLLoss()
    else:
        raise NameError("Unknown loss function.")

    # Run each optimizer on the problem
    prob_confs = conf_dict["problem_configs"]

    torch.manual_seed(seed)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = "33068"
    for prob_key in prob_confs:
        prob_conf = prob_confs[prob_key]
        opt_conf = prob_conf["optimizer_config"]

        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Running problem: " + prob_conf["problem_name"])
        # torch.backends.cudnn.deterministic = True
        mp.spawn(worker, nprocs=N,args=(graph,model_conf,prob_conf,opt_conf,train_subsets,val_set,output_dir))

  

        # if exp_conf["writeout"]:
        #     prob.save_metrics(output_dir)


    return 

def worker(i,graph,model_conf,prob_conf,opt_conf,train_subsets,val_set,output_dir):
    
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:33068", rank=i, world_size=graph.number_of_nodes())
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            num_gpu = torch.cuda.device_count()
            torch.cuda.set_device(i % num_gpu)
            
        base_model = MNISTConvNet(
            model_conf["num_filters"],
            model_conf["kernel_size"],
            model_conf["linear_width"],
            )
        pr=DistMNISTProblem_Single(
            graph,
            base_model,
            torch.nn.NLLLoss(),
            train_subsets[i],
            val_set,
            device,
            prob_conf,
        )
        parameter=list(pr.model.parameters())
        parameter_2=torch.nn.utils.parameters_to_vector(parameter).clone().detach()
        num_params = len(parameter)
        alpha=opt_conf["alpha"]
        tau=opt_conf["tau"]
        use_prox=opt_conf["use_prox"]
        quant_bit=opt_conf["quantize"]
        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=device)
            for p in parameter
        ]
        g_estimator=copy.deepcopy(base_zeros)
        y_estimator=copy.deepcopy(base_zeros)

        eval_every = pr.conf["metrics_config"]["evaluate_frequency"]
        oits = opt_conf["outer_iterations"]
        if opt_conf["lr_decay_type"] == "constant":
            primal_lr = opt_conf["primal_lr_start"] * torch.ones(
                opt_conf["outer_iterations"]
            )
        elif opt_conf["lr_decay_type"] == "linear": 
            primal_lr = torch.linspace(
                opt_conf["primal_lr_start"],
                opt_conf["primal_lr_finish"],
                opt_conf["outer_iterations"],
            )
        elif opt_conf["lr_decay_type"] == "log":
            primal_lr = torch.logspace(
                math.log(opt_conf["primal_lr_start"], 10),
                math.log(opt_conf["primal_lr_finish"], 10),
                opt_conf["outer_iterations"],
            )
        else:
            raise NameError("Unknow primal learning rate decay type.")
        pits = opt_conf["primal_iterations"]

        # Initialize Ylists and Glists
        if opt_conf["init_grads"]:
            bloss = pr.local_batch_loss()
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
        W = graph_generation.get_metropolis(pr.graph).to(device)
        graph=My_Graph(rank=i,world_size=pr.N,weight_matrix=W)
        out_edges, in_edges = graph.get_edges()
        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                # print("rank: ",i," iteration: ",k)
                if i==0:
                    pr.evaluate_metrics(at_end=(k == oits - 1))
                
                
                    
                
            pr.update_graph()

            # # Compute graph weights
            W = graph_generation.get_metropolis(pr.graph)
            W = W.to(device)
            
            
            parameter_2=local_update(i,primal_lr[k],parameter,parameter_2,y_estimator,g_estimator,pr,opt_conf)
            
            # neighs = list(pr.graph.neighbors(i))
            # send_msg=parameter_2.clone().detach()
            # for j in neighs:
            #     weight=W[j,i].to(device)
            #     dist.send(send_msg.clone().detach().mul(weight),dst=j)
            # in_msg=W[i,i]*parameter_2
            # recv_msg=torch.zeros_like(parameter_2)
            # for j in neighs:
            #     dist.recv(tensor=recv_msg,src=j)
            #     in_msg.add_(recv_msg)
            
            send_msg=parameter_2.clone().detach()
            for out_edge in out_edges:
                assert i==out_edge.src
                weight=W[out_edge.dest,i].to(device)
                dist.broadcast(send_msg.mul(weight),src=i,group=out_edge.process_group,async_op=True)
            in_msg=W[i,i]*parameter_2
            recv_msg=torch.zeros_like(parameter_2)
            for in_edge in in_edges:
                dist.broadcast(tensor=recv_msg,src=in_edge.src,group=in_edge.process_group)
                in_msg.add_(recv_msg)
                
            # print("rank: ",i," in_msg: ",in_msg)
            # Iterate over the agents for communication step
            torch.nn.utils.vector_to_parameters(in_msg,parameter)
   
        
            bloss = pr.local_batch_loss()
            bloss.backward()
            
            # y_send_msg=torch.nn.utils.parameters_to_vector(y_estimator).clone().detach()
            # for j in neighs:
            #     weight=W[j,i].to(device)
            #     dist.send(y_send_msg.clone().detach().mul(weight),dst=j)
            # in_y_msg=W[i,i]*torch.nn.utils.parameters_to_vector(y_estimator)
            
            # recv_y_msg=torch.zeros_like(y_send_msg)
            # for j in neighs:
            #     dist.recv(tensor=recv_y_msg,src=j)
            #     in_y_msg.add_(recv_y_msg)
            # torch.nn.utils.vector_to_parameters(in_y_msg,y_estimator)
            
            # Compute the batch loss and update using the gradients
            y_send_msg=torch.nn.utils.parameters_to_vector(y_estimator).clone().detach()
            for out_edge in out_edges:
                assert i==out_edge.src
                weight=W[out_edge.dest,i].to(device)
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

            dist.barrier()
        pr.save_metrics(i,output_dir)
        return       
    
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
        pred_loss = pr.local_batch_loss()
        theta=torch.nn.utils.parameters_to_vector(parameter)
        surrogate_loss=conf["tau"]/2*torch.square(torch.norm(theta-ori_theta))
        loss=pred_loss+surrogate_loss+torch.dot(delta_grad,theta)
        loss.backward()
        opt.step()
    opt.zero_grad()
    alpha=conf["alpha"]
    parameter_2=(1-alpha)*ori_theta+alpha*torch.nn.utils.parameters_to_vector(parameter).detach().clone()
    return parameter_2

if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")
