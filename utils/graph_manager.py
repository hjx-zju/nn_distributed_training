"""
==========================
@author:Zhu Zehan
@time:2020/11/25:19:48
@email:12032045@zju.edu.cn
==========================
"""
import torch.distributed as dist
import torch
from datetime import timedelta

class Edge(object):
    """
    定义两个节点之间构成一个通信子进程组
    """
    def __init__(self, dest, src):
        self.src = src
        self.dest = dest
        self.process_group = dist.new_group([src, dest],timeout=timedelta(seconds=10))


class GraphManager(object):

    def __init__(self, rank, world_size, weight_matrix=None):
        self.rank = rank
        self.world_size = world_size
        self.phone_book = [[] for _ in range(self.world_size)]
        self.weight_matrix = weight_matrix
        self.out_peers = weight_matrix[:, rank].nonzero().flatten().tolist()
        self.out_peers.remove(rank)
        self.in_peers = weight_matrix[rank, :].nonzero().flatten().tolist()
        self.in_peers.remove(rank)
        self._make_graph()

    def _make_graph(self):
        """
        """
        raise NotImplementedError

    def _add_peers(self, rank, peers):
        for peer in peers:
            self.phone_book[rank].append(Edge(dest=peer, src=rank))

    def get_edges(self):
        """
        得到外向邻居列表、内向邻居列表
        """
        # get out- and in-peers using new group-indices
        out_edges = self.phone_book[self.rank]

        in_edges = []
        for rank, edges in enumerate(self.phone_book):
            if rank == self.rank:
                continue
            for edge in edges:
                if self.rank == edge.dest:
                    in_edges.append(edge)

        return out_edges, in_edges


class My_Graph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            out_peers = self.weight_matrix[:, rank].nonzero().flatten().tolist()
            out_peers.remove(rank)
            self._add_peers(rank, out_peers)

if __name__ == '__main__':
    weight_matrix = torch.tensor([[0.5,0.5],
                                  [0.5,0.5]
                                 ])
    # graph = My_Graph(0, 2, weight_matrix)
    # out_edges, in_edges = graph.get_edges()
    # print(out_edges)
    # print(in_edges)
    # print(graph.out_peers)
    # print(graph.in_peers)
    # print(graph.phone_book)
    # print(weight_matrix[:, 0].nonzero().flatten().tolist())
    # print(torch.distributed.is_gloo_available())