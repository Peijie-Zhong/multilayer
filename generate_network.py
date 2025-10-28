import math
import random
from typing import List, Dict, Set, Tuple
import csv
import numpy as np
import multilayerGM as gm  # 来自仓库


def build_layer_node_sets(
    N: int,
    L: int,
    init_size: int,
    join_rate: float,
    leaving_rate: float,
    *,
    stochastic: str = "binomial",
    seed: int = 42,
):
    assert 0 <= init_size <= N
    assert 0 <= join_rate <= 1 and 0 <= leaving_rate <= 1
    assert L >= 1

    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    universe = list(range(N))
    layer_sets: List[Set[int]] = []

    S0 = set(py_rng.sample(universe, init_size))
    layer_sets.append(S0)

    seen = set(S0) 
    stats = [{
        "layer": 0,
        "size": len(S0),
        "survivors": len(S0),  
        "entrants": len(S0),   
        "jaccard_with_prev": 1.0  
    }]

    for l in range(1, L):
        prev = layer_sets[-1]
        a = len(prev)
        survive_prob = 1.0 - leaving_rate
        survivors = {u for u in prev if rng.random() < survive_prob}
        S_count = len(survivors)
        candidates = [u for u in universe if (u not in survivors)]
        if stochastic == "binomial":
            n_join = int(rng.binomial(n=len(prev), p=join_rate))
        elif stochastic == "expected":
            n_join = int(round(join_rate * a))
        else:
            raise ValueError("stochastic must be 'binomial' or 'expected'")

        n_join = min(n_join, len(candidates)) 

        entrants = set(py_rng.sample(candidates, n_join)) if n_join > 0 else set()
        Sl = survivors | entrants

        layer_sets.append(Sl)
        seen |= entrants

        # 统计
        inter = len(prev & Sl)
        union = len(prev | Sl)
        jacc = (inter / union) if union > 0 else 0.0

        stats.append({
            "layer": l,
            "size": len(Sl),
            "survivors": S_count,
            "entrants": n_join,
            "jaccard_with_prev": jacc
        })

    return layer_sets, stats


def sample_layer_labels_inherit(
    layer_sets: List[Set[int]],
    n_sets: int = 4,
    theta: float = 1.0,
    p_stay: float = 0.8, 
    seed: int = 42,
) -> Dict[Tuple[int, int], int]:

    rng = np.random.default_rng(seed)
    partition: Dict[Tuple[int, int], int] = {}

    probs0 = rng.dirichlet(alpha=[theta] * n_sets)
    for u in layer_sets[0]:
        partition[(u, 0)] = int(rng.choice(n_sets, p=probs0))

    for l in range(1, len(layer_sets)):
        prev_nodes = layer_sets[l - 1]
        cur_nodes  = layer_sets[l]
        probs_l = rng.dirichlet(alpha=[theta] * n_sets)

        survivors = cur_nodes & prev_nodes
        entrants  = cur_nodes - prev_nodes

        for u in survivors:
            if rng.random() < p_stay:
                partition[(u, l)] = partition[(u, l - 1)]
            else:
                partition[(u, l)] = int(rng.choice(n_sets, p=probs_l))

        for u in entrants:
            partition[(u, l)] = int(rng.choice(n_sets, p=probs_l))

    return partition


def generate_multilayer_network(
    node_pool: int,
    layer: int, 
    init_size: int,
    n_sets: int,
    mu: int, 
    k_min: int, 
    k_max: int, 
    t_k: int,
    theta:int, 
    p_stay: int, 
    join_rate: int = 0.0,
    leaving_rate: int = 0.0,
    stochastic: str = "binomial",
    seed: int = 42, 
    output_path: str = None
):
    layer_sets, stats = build_layer_node_sets(
        N=node_pool, L=layer, init_size=init_size,
        join_rate=join_rate, 
        leaving_rate=leaving_rate,
        stochastic=stochastic,  
        seed=seed
    )
    print(stats)
    partition = sample_layer_labels_inherit(layer_sets, n_sets=n_sets, theta=theta, seed=seed, p_stay=p_stay)

    multinet = gm.multilayer_DCSBM_network(partition, mu=mu, k_min=k_min, k_max=k_max, t_k=t_k)
    allowed = set(partition.keys())
    to_remove = [n for n in list(multinet.nodes) if n not in allowed]
    if to_remove:  
        multinet.remove_nodes_from(to_remove)
    for n in multinet.nodes:
        multinet.nodes[n]['mesoset'] = partition[n]

    if output_path != None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["u", "v", "layer", "u_label", "v_label"])
            for (u_node, v_node) in multinet.edges():
                if u_node[1] != v_node[1]:
                    continue
                u_phys, layer_u = u_node[0], u_node[1]
                v_phys, _ = v_node[0], v_node[1]
                u_lab = multinet.nodes[u_node].get("mesoset", -1)
                v_lab = multinet.nodes[v_node].get("mesoset", -1)
                writer.writerow([u_phys, v_phys, layer_u, u_lab, v_lab])