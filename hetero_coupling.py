from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from ot.gromov import gromov_wasserstein, gromov_wasserstein2, entropic_gromov_wasserstein, entropic_gromov_wasserstein2

def read_edges(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"u", "v", "layer"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing required columns: {sorted(miss)}")
    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)
    df["layer"] = df["layer"].astype(str)
    return df

def build_layer_graphs(df: pd.DataFrame) -> Dict[str, nx.Graph]:
    graphs: Dict[str, nx.Graph] = {}
    for layer, gdf in df.groupby("layer"):
        G = nx.Graph()
        if "w" in gdf.columns:
            edges = [(u, v, float(w)) for u, v, w in gdf[["u", "v", "w"]].itertuples(index=False)]
            G.add_weighted_edges_from(edges)
        else:
            G.add_edges_from(gdf[["u", "v"]].itertuples(index=False, name=None))
            nx.set_edge_attributes(G, 1, "weight")
        for col in ["u", "v"]:
            for n in gdf[col].unique():
                if n not in G:
                    G.add_node(n)
        graphs[layer] = G
    return graphs

def compute_node2vec_embeddings(
    G: nx.Graph,
    dimensions=64, walk_length=80, num_walks=10, p=1.0, q=1.0,
    window=10, workers=1, seed=42
) -> pd.DataFrame:
    if len(G) == 0:
        raise ValueError("Empty graph.")
    def _fit(n_jobs):
        n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                       num_walks=num_walks, p=p, q=q, workers=n_jobs,
                       seed=seed, quiet=True)
        return n2v.fit(window=window, min_count=1, batch_words=4)
    try:
        model = _fit(workers)
    except Exception as e:
        print(f"[WARN] node2vec workers={workers} failed: {e}. Fallback to workers=1.")
        model = _fit(1)
    nodes = [str(n) for n in G.nodes()]
    X = np.vstack([model.wv[n] for n in nodes])
    emb = pd.DataFrame(X, index=nodes, columns=[f"f{i}" for i in range(X.shape[1])])
    emb.index.name = "node"
    return emb

def pairwise_from_embeddings(emb: pd.DataFrame, metric="euclidean") -> np.ndarray:
    X = emb.values.astype(float)
    if metric == "euclidean":
        s = np.sum(X * X, axis=1, keepdims=True)
        D2 = s + s.T - 2 * (X @ X.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, dtype=float)
    elif metric == "cosine":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Xn = X / n
        S = np.clip(Xn @ Xn.T, -1.0, 1.0)
        return 1.0 - S
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'.")
    
def gw_coupling_and_distance(emb1: pd.DataFrame, emb2: pd.DataFrame, metric="euclidean"):
    emb1 = emb1.sort_index()
    emb2 = emb2.sort_index()
    nodes1 = emb1.index.to_list()
    nodes2 = emb2.index.to_list()
    C1 = pairwise_from_embeddings(emb1, metric=metric)
    C2 = pairwise_from_embeddings(emb2, metric=metric)
    n1, n2 = C1.shape[0], C2.shape[0]
    p = np.ones(n1) / max(n1, 1)
    q = np.ones(n2) / max(n2, 1)
    # π（最优耦合，带熵正则版可用 ot.gromov.entropic_gromov_wasserstein）
    pi = entropic_gromov_wasserstein(C1, C2, p, q, loss_fun="square_loss")
    # GW^2
    gw2 = entropic_gromov_wasserstein2(C1, C2, p, q, loss_function="square_loss")
    gw = float(np.sqrt(max(gw2, 0.0)))
    return pi, nodes1, nodes2, gw