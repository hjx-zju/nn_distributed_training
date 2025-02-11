# 安装

- 环境安装参考[readme.md](./README.md)

# 运行

- yaml文件的相关说明可以参考[readme.md](./README.md)

```python
# Starting in repo root, and assuming you have
# the virtualenv configured
> source venv/bin/activate
> cd experiments/
> python [experiment script name] [path to yaml file]

# mnist 实验:
> python dist_mnist_ex.py dist_mnist_PAPER.yaml
# mnist scaling节点数量实验：
> python dist_mnist_scaling.py dist_mnist_scaling.yaml

# 建图实验
> python dist_online_dense_ex.py dist_online_dense_PAPER.yaml

# 带有anim字样的yaml文件通常会额外存储一些metric信息，用于可视化

# RL实验在RL/文件夹下
#集中式训练，结果保存在results_cen/中
> python main.py
#分布式训练，在RL/dist_rl文件夹下,运行对应算法的训练文件。算法实现一般在 {alg}_PPO.py中
> python train_{alg}_multi.py

```



# 可视化

- MNIST和建图可视化结果在visualization文件夹下的mnist_four.ipynb和online_density_minimal_overlap.ipynb中。

- RL的可视化文件为RL/dist_rl/test.ipynb