import math
import random
from typing import List, Dict, Set, Tuple
import csv

import numpy as np
import multilayerGM as gm  # 来自仓库


def jaccard_intersection_size(a: int, b: int, c: float) -> int:
    """
    给定集合大小 a、b 和目标 Jaccard c，解 |A∩B| = x，使 x/(a+b-x)=c
    x = c*(a+b)/(1+c)，并限制在 [0, min(a,b)]
    """
    if c < 0 or c > 1:
        raise ValueError("coverage 中的值必须在 [0,1] 之间")
    if a < 0 or b < 0:
        raise ValueError("nodes_per_layer 必须为非负整数")

    if a == 0 or b == 0:
        return 0

    if c == 0:
        return 0
    # 理想值
    x_float = (c * (a + b)) / (1.0 + c)
    # 四舍五入后再裁剪
    x = int(round(x_float))
    x = max(0, min(x, min(a, b)))
    return x


def build_layer_node_sets(
    N: int, coverage: List[float], nodes_per_layer: List[int], seed: int = 42
) -> List[Set[int]]:
    """
    在节点池 [0, N-1] 里为每层挑选节点集合 S_l
    使得相邻层 l 与 l+1 的 Jaccard 近似于 coverage[l]
    """
    rng = random.Random(seed)
    L = len(nodes_per_layer)
    if len(coverage) != L - 1:
        raise ValueError("coverage 的长度必须等于 len(nodes_per_layer)-1")

    if any(n < 0 for n in nodes_per_layer):
        raise ValueError("nodes_per_layer 里不能有负数")
    if any(n > N for n in nodes_per_layer):
        raise ValueError("每层的节点数不能超过节点池大小 N")

    universe = list(range(N))

    # 先随机取第一层
    S = []
    S0 = set(rng.sample(universe, nodes_per_layer[0]))
    S.append(S0)

    # 逐层构造
    for l in range(1, L):
        a = nodes_per_layer[l - 1]
        b = nodes_per_layer[l]
        c = coverage[l - 1]

        prev = S[l - 1]
        # 计算目标交集大小
        x = jaccard_intersection_size(a, b, c)
        # 与前一层交集从 prev 里抽
        if x > len(prev):
            x = len(prev)
        inter = set(rng.sample(list(prev), x))

        # 剩余需要的新元素数
        rest = b - x

        # 优先从未在 prev 中的节点取，避免影响 Jaccard
        candidates = [u for u in universe if u not in prev]
        if len(candidates) >= rest:
            add = set(rng.sample(candidates, rest))
        else:
            # 如果候选不够，就允许从 prev 之外的“已使用过但不在 prev 的层”里取
            used = set().union(*S) if S else set()
            more_candidates = [u for u in universe if (u not in inter)]
            # 保证最终大小为 b
            add = set()
            for u in rng.sample(more_candidates, b - len(inter)):
                add.add(u)
                if len(inter) + len(add) == b:
                    break

        Sl = inter | add
        # 兜底修正（极端情况下可能超或少一两个）
        if len(Sl) > b:
            Sl = set(list(Sl)[:b])
        if len(Sl) < b:
            extra = [u for u in universe if u not in Sl]
            Sl |= set(rng.sample(extra, b - len(Sl)))

        S.append(Sl)

    return S


def sample_layer_labels(
    layer_sets: List[Set[int]],
    n_sets: int = 4,
    theta: float = 1.0,
    seed: int = 42,
) -> Dict[Tuple[int, int], int]:
    """
    为每个状态节点 (node, layer) 采样一个“社区/mesoset”标签
    使用对称 Dirichlet(θ) 产生层内类别分布，然后分类采样
    """
    rng = np.random.default_rng(seed)
    partition: Dict[Tuple[int, int], int] = {}

    for l, nodes in enumerate(layer_sets):
        # Dirichlet 概率
        probs = rng.dirichlet(alpha=[theta] * n_sets)
        # 为该层的每个节点抽签
        for u in nodes:
            label = rng.choice(n_sets, p=probs)
            partition[(u, l)] = int(label)

    return partition


def build_multilayer_network(
    partition: Dict[Tuple[int, int], int],
    mu: float = 0.1,
    k_min: int = 5,
    k_max: int = 70,
    t_k: float = -2.0,
):
    """
    用 MultilayerGM 的 DCSBM 基准模型按给定 partition 生成多层网络
    返回 MultilayerGraph（节点形如 (u, layer)），并带 'mesoset' 属性
    """
    # gm.multilayer_DCSBM_network 会读取节点上的 'mesoset'（我们会在内部设置）
    # 需要把 partition 转换为节点属性字典：
    # 其内部会创建节点并设置 'mesoset'，我们只需传 mapping 即可。
    multinet = gm.multilayer_DCSBM_network(
        partition, mu=mu, k_min=k_min, k_max=k_max, t_k=t_k
    )
    return multinet


def export_edges_csv(
    multinet,
    out_path: str = "edges.csv",
):
    """
    将多层网络的“同层边”导出为 CSV：
    u,v,layer,u_label,v_label
    其中 layer 从 0 开始计数；u_label/v_label 取自节点属性 'mesoset'
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "layer", "u_label", "v_label"])

        # multinet 的节点是 (u, layer) 的元组
        for (u_node, v_node) in multinet.edges():
            # 只导出“同层边”（一般 DCSBM 生成的就是同层边）
            if u_node[1] != v_node[1]:
                continue
            u_phys, layer_u = u_node[0], u_node[1]
            v_phys, layer_v = v_node[0], v_node[1]
            # 取标签
            u_lab = multinet.nodes[u_node].get("mesoset", -1)
            v_lab = multinet.nodes[v_node].get("mesoset", -1)
            writer.writerow([u_phys, v_phys, layer_u, u_lab, v_lab])

