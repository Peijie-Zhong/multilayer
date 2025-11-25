import pandas as pd
from infomap import Infomap
import numpy as np
from typing import Dict, List, Tuple
from scipy import sparse
import matlab.engine
import matlab
from hetero_coupling import *

# ===================Infomap===================
def infomap_communities(input_csv_path, output_csv_path, jsd_relax_rate=0.25):
    df = pd.read_csv(input_csv_path)
    im = Infomap()
    node_labels = pd.concat([df['u'], df['v']]).unique()
    layer_labels = df['layer'].unique()
    node_map = {label: i for i, label in enumerate(node_labels)}
    layer_map = {label: i for i, label in enumerate(layer_labels)}
    rev_node_map = {v: k for k, v in node_map.items()}
    rev_layer_map = {v: k for k, v in layer_map.items()}
    for _, row in df.iterrows():
        u_id = node_map[row['u']]
        v_id = node_map[row['v']]
        layer_id = layer_map[row['layer']]
        im.add_multilayer_intra_link(layer_id, u_id, v_id)
    im.run(two_level=True,
           multilayer_relax_by_jsd=True,
           multilayer_relax_rate=jsd_relax_rate,
           multilayer_relax_limit=1,
           silent=True)
    results = []
    for node in im.nodes:
        physical_node_id = node.node_id
        layer_id = node.layer_id
        community_id = node.module_id
        results.append({
            'node_id': rev_node_map[physical_node_id],
            'layer': rev_layer_map[layer_id],
            'community': community_id
        })
    
    output_df = pd.DataFrame(results).sort_values(['layer', 'node_id']).reset_index(drop=True)
    if output_csv_path != None:    
        output_df.to_csv(output_csv_path, index=False)
        print(f"results write to {output_csv_path}.")
    return output_df

# ===================GenLouvain===================
def build_multilayer_adjacency(
    df: pd.DataFrame,
    directed: bool = False,
    weight: bool = False,
) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
    lower_map = {c.lower(): c for c in df.columns}
    for col in ("u", "v", "layer"):
        if col not in lower_map:
            raise ValueError(
                f"Missing required column '{col}'. Found columns: {list(df.columns)}"
            )
    df = df.rename(columns={
        lower_map["u"]: "u",
        lower_map["v"]: "v",
        lower_map["layer"]: "layer"
    })

    df = df.dropna(subset=["u", "v", "layer"]).copy()
    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)

    # --- weight 检查 ---
    if weight:
        if "weight" not in df.columns:
            raise ValueError(
                "use_weight=True but no column named 'weight' was found in df."
            )
        # 转 float，确保矩阵可用
        df["weight"] = df["weight"].astype(float)

    # Sort layers numerically if possible
    try:
        df["_layer_num"] = pd.to_numeric(df["layer"], errors="raise")
        layers_sorted_num = sorted(df["_layer_num"].unique().tolist())
        layer_text_for_num = {
            ln: str(df.loc[df["_layer_num"] == ln, "layer"].iloc[0])
            for ln in layers_sorted_num
        }
        layers_out = [layer_text_for_num[ln] for ln in layers_sorted_num]
        layer_key = "_layer_num"
    except Exception:
        layers_out = sorted(df["layer"].astype(str).unique().tolist())
        layer_key = "layer"

    # Node universe
    nodes = sorted(set(df["u"]).union(set(df["v"])), key=lambda x: (len(x), x))
    node_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    A_list: List[np.ndarray] = []
    for L in layers_out:
        if layer_key == "_layer_num":
            numeric = [k for k, v in layer_text_for_num.items() if v == L][0]
            sub = df[df["_layer_num"] == numeric]
        else:
            sub = df[df["layer"].astype(str) == L]

        if sub.empty:
            A_list.append(np.zeros((N, N), dtype=float))
            continue

        ui = sub["u"].map(node_index).to_numpy()
        vi = sub["v"].map(node_index).to_numpy()

        # --- 核心：根据 use_weight 决定 data 是 1 还是来自 weight ---
        if weight:
            data = sub["weight"].to_numpy(dtype=float)
        else:
            data = np.ones(len(sub), dtype=float)

        A = sparse.coo_matrix((data, (ui, vi)), shape=(N, N))

        if not directed:
            A = A + A.T
            A.setdiag(0)
            A.eliminate_zeros()

        A_list.append(A.toarray())

    return A_list, layers_out, node_index

def call_genlouvain(
    A_list: List[np.ndarray],
    genlouvain_path: str,
    omega: float = 1.0,
    gamma: float | List[float] | np.ndarray = 1.0,
    algorithm: str = "naive",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Call MATLAB's GenLouvain on a multilayer network using 'multiord' (preferred)
    or fallback 'multislice' to construct the modularity matrix.

    Returns:
        S_mat: (N x T) numpy array of community assignments (1-indexed as in GenLouvain)
    """
    eng = matlab.engine.start_matlab()

    # Add GenLouvain and subfolders to MATLAB path
    _ = eng.addpath(eng.genpath(genlouvain_path), nargout=1)

    # Convert A_list to MATLAB cell array of double matrices
    T = len(A_list)
    if T == 0:
        raise ValueError("No layers provided to GenLouvain.")
    N = int(A_list[0].shape[0])

        # Build MATLAB cell array as a Python list (Engine auto-converts lists to cell arrays)
    A_cell = []
    for t in range(T):
        Ai = np.array(A_list[t], dtype=float, copy=False)
        np.fill_diagonal(Ai, 0.0)
        A_cell.append(matlab.double(Ai.tolist()))

    # gamma: scalar or per-layer vector
    if np.isscalar(gamma):
        gamma_ml = float(gamma)
    else:
        gamma_vec = [float(g) for g in np.ravel(gamma).tolist()]
        if len(gamma_vec) != T:
            raise ValueError(f"gamma must be a scalar or length {T} vector.")
        import matlab as _ml  # type: ignore
        gamma_ml = _ml.double(gamma_vec)
    omega = float(omega)

    # Put variables into MATLAB workspace
    eng.workspace['A_cell'] = A_cell
    eng.workspace['omega'] = float(omega)
    eng.workspace['gamma_ml'] = gamma_ml

    # Compute B (sparse) in MATLAB and keep it there
    eng.eval('[B, twom] = multiord(A_cell, gamma_ml, omega);', nargout=0)

    if random_state is not None:
        eng.eval(f"rng({int(random_state)});", nargout=0)

    # Run the community detection in MATLAB, only fetch S and Q (dense)
    has_iter = int(eng.feval('exist', 'iterated_genlouvain', 'file', nargout=1)) == 2
    if algorithm == "naive":
        eng.eval('[S, Q] = genlouvain(B);', nargout=0)
    elif algorithm == "iterated":
        eng.eval('[S, Q] = iterated_genlouvain(B, [], [], 0);', nargout=0)
    elif algorithm == "itermodmax":
        eng.addpath(eng.genpath("IterModMax"), nargout=0)
        eng.eval('[gamma, omega, ~, S] = it_mod_max_temporal(A_cell, gamma_ml, omega);', nargout=0)


    S = eng.workspace['S']

    # Convert S (length N*T vector in MATLAB) to N x T numpy array
    S_py = np.array(S).ravel(order="F")
    if S_py.size != N * T:
        S_py = np.array(list(np.array(S).flatten()), dtype=float).reshape((N * T,), order="F")
    S_mat = S_py.reshape((N, T), order="F")
    eng.quit()
    return S_mat


def genlouvain_communities(
    input_data: str | pd.DataFrame,
    output_csv: str = None,
    omega: float = 1.0,
    gamma: float | List[float] | np.ndarray = 1.0,
    directed: bool = False,
    weight: bool = False,
    random_state: int | None = None,
    algorithm: str = "naive",
) -> None:
    if type(input_data) == str:
        df = pd.read_csv(input_data)
    else:
        df = input_data

    A_list, layers, node_index = build_multilayer_adjacency(df, directed=directed, weight=weight)

    S = call_genlouvain(
        A_list=A_list,
        genlouvain_path="GenLouvain",
        omega=omega,
        gamma=gamma,
        algorithm=algorithm,
        random_state=random_state        
    )

    # Prepare output rows
    nodes_by_index = {i: n for n, i in node_index.items()}
    N, T = S.shape
    rows = []
    for t_idx, layer in enumerate(layers):
        for i in range(N):
            rows.append({
                "node_id": nodes_by_index[i],
                "layer": str(layer),
                "community": int(S[i, t_idx]),
            })
    out_df = pd.DataFrame(rows, columns=["node_id", "layer", "community"])
    out_df = out_df.sort_values(['layer', 'node_id']).reset_index(drop=True)
    if output_csv != None:
        out_df.to_csv(output_csv, index=False)
    return out_df


# =================== Hetero GenLouvain ===================

def layer_adjacency_matrix(G: nx.Graph, node_order: List[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_order)}
    N = len(node_order)
    rows = []
    cols = []
    vals = []
    for u, v, data in G.edges(data=True):
        if u not in idx or v not in idx:
            continue
        w = data.get("weight", 1.0)
        i = idx[u]
        j = idx[v]
        rows.append(i); cols.append(j); vals.append(w)
        rows.append(j); cols.append(i); vals.append(w)
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).toarray()
    np.fill_diagonal(A, 0.0)
    return A

def supra_adjacency_from_pi(
    graphs: Dict[str, nx.Graph],
    embeddings: Dict[str, pd.DataFrame],
    metric: str = "euclidean",
    omega_scale: float = 1.0,
) -> Tuple[np.ndarray, List[Tuple[str, str]], Dict[str, List[str]]]:

    layers_sorted = sorted(graphs.keys())

    # 1. 为每一层确定节点顺序（优先用该层 embedding 的 index）
    layer_nodes: Dict[str, List[str]] = {}
    for layer in layers_sorted:
        nodes_emb = [str(n) for n in embeddings[layer].index.tolist()]
        extra = [str(n) for n in graphs[layer].nodes() if str(n) not in nodes_emb]
        full_nodes = nodes_emb + extra
        # 去重保持顺序
        seen = set()
        ordered_nodes = []
        for n in full_nodes:
            if n not in seen:
                seen.add(n)
                ordered_nodes.append(n)
        layer_nodes[layer] = ordered_nodes

    # 2. 每层内邻接块
    layer_adj_blocks: Dict[str, np.ndarray] = {}
    for layer in layers_sorted:
        layer_adj_blocks[layer] = layer_adjacency_matrix(
            graphs[layer],
            layer_nodes[layer]
        )

    # 3. 只计算“相邻层(t, t+1)”的跨层块
    #    coupling_blocks[(la, lb)] = 矩阵 (|V_la| x |V_lb|)
    coupling_blocks: Dict[Tuple[str, str], np.ndarray] = {}

    for k in range(len(layers_sorted) - 1):
        la = layers_sorted[k]
        lb = layers_sorted[k + 1]

        emb_a = embeddings[la]
        emb_b = embeddings[lb]

        # pi: shape (|V_la'| x |V_lb'|)
        # nodes_a, nodes_b 对应 pi 的行/列索引
        pi, nodes_a, nodes_b, gw_dist = gw_coupling_and_distance(
            emb_a, emb_b, metric=metric
        )

        # 我们需要把 pi 的行/列顺序对齐到 layer_nodes[la], layer_nodes[lb]
        idx_a = {n: i for i, n in enumerate(nodes_a)}
        idx_b = {n: j for j, n in enumerate(nodes_b)}

        na = len(layer_nodes[la])
        nb = len(layer_nodes[lb])

        C = np.zeros((na, nb), dtype=float)

        # 把 GW 耦合强度映射到 (la节点顺序, lb节点顺序)
        for ia, node_a in enumerate(layer_nodes[la]):
            ka = idx_a.get(node_a, None)
            if ka is None:
                continue
            row_pi = pi[ka, :]  # pi[ka, :] 对应 node_a -> 所有 node_b
            for ib, node_b in enumerate(layer_nodes[lb]):
                kb = idx_b.get(node_b, None)
                if kb is None:
                    continue
                C[ia, ib] = row_pi[kb]

        if omega_scale != 1.0:
            C = C * float(omega_scale)

        # 保存 t -> t+1 的块 和 对称的 t+1 -> t
        coupling_blocks[(la, lb)] = C
        coupling_blocks[(lb, la)] = C.T

    # 4. 把这些块拼成一个大矩阵
    # 4a. 给每层一个 offset，方便把子块放进总矩阵
    layer_offset: Dict[str, int] = {}
    offset = 0
    for layer in layers_sorted:
        layer_offset[layer] = offset
        offset += len(layer_nodes[layer])
    total_N = offset

    A_supra = np.zeros((total_N, total_N), dtype=float)

    # 4b. 放每层内部邻接 (对角块)
    for layer in layers_sorted:
        A_block = layer_adj_blocks[layer]
        base = layer_offset[layer]
        na = A_block.shape[0]
        if na == 0:
            continue
        A_supra[base:base+na, base:base+na] = A_block

    # 4c. 放相邻层之间的耦合块
    for (la, lb), C_ab in coupling_blocks.items():
        base_a = layer_offset[la]
        base_b = layer_offset[lb]
        na, nb = C_ab.shape
        if na == 0 or nb == 0:
            continue
        A_supra[base_a:base_a+na, base_b:base_b+nb] = C_ab

    # 5. global_index: A_supra 的行/列对应哪个 (layer,node)
    global_index: List[Tuple[str, str]] = []
    for layer in layers_sorted:
        for node in layer_nodes[layer]:
            global_index.append((layer, node))

    return A_supra, global_index, layer_nodes, layer_adj_blocks, coupling_blocks

def modularity_matrix(A: np.ndarray) -> np.ndarray:
    k = A.sum(axis=1, keepdims=True)  # shape (N,1)
    two_m = float(k.sum())            # scalar = sum of all weights
    if two_m <= 0:
        # 没边的情况下，B 就是 0 矩阵
        return np.zeros_like(A, dtype=float)

    expected = (k @ k.T) / two_m      # outer product / (2m)
    B = A - expected
    # 可选：去掉非常小的数值噪声
    B[np.abs(B) < 1e-15] = 0.0
    return B


def build_B_from_blocks(
    layer_adj_blocks: Dict[str, np.ndarray],
    coupling_blocks: Dict[Tuple[str, str], np.ndarray],
    layer_nodes: Dict[str, List[str]],
    gamma: float = 1.0,
) -> np.ndarray:
    layers = sorted(layer_nodes.keys())
    layer_offset = {}
    offset = 0
    for layer in layers:
        layer_offset[layer] = offset
        offset += len(layer_nodes[layer])
    N = offset

    # 2) 初始化 B 为 0
    B = np.zeros((N, N), dtype=float)

    # 3) 对角块：每层独立计算 expected_s = (k_s k_s^T) / (2 m_s)
    for layer in layers:
        A = layer_adj_blocks.get(layer)
        if A is None:
            # 空层（无节点）
            continue
        base = layer_offset[layer]
        n = A.shape[0]
        # 层内度与2m_s
        k_s = A.sum(axis=1, keepdims=True)        # (n,1)
        two_m_s = float(k_s.sum())                # scalar
        if two_m_s > 0:
            expected_s = (k_s @ k_s.T) / two_m_s
        else:
            expected_s = np.zeros_like(A)
        B_block = A - gamma * expected_s
        # 写进 B 的对角块
        B[base:base+n, base:base+n] = B_block

    # 4) 添加跨层耦合块（直接加入 B 的 off-diagonal）
    #    coupling_blocks[(la,lb)] 形状应为 (len(layer_nodes[la]), len(layer_nodes[lb]))
    for (la, lb), C in coupling_blocks.items():
        if la not in layer_offset or lb not in layer_offset:
            # 忽略未知层（安全检查）
            continue
        base_a = layer_offset[la]
        base_b = layer_offset[lb]
        na = C.shape[0]
        nb = C.shape[1]
        # 在 B 的对应位置直接加上 C（可能是稀疏的）
        # 如果 (lb,la) 也在 coupling_blocks，会在其迭代时加对称项两次——
        # 因此，期望 coupling_blocks[(lb,la)] == C.T，或只在上三角加入并手动对称填充。
        B[base_a:base_a+na, base_b:base_b+nb] += C
    # 5) 对称化（保证数值对称）
    B = (B + B.T) / 2.0
    # 清理数值噪声
    B[np.abs(B) < 1e-15] = 0.0
    return B


import matlab.engine
import matlab
def run_hetero_genlouvain(
    B: np.ndarray,
    matlab_engine: matlab.engine,
    random_state: int | None = None,
    use_iterated: bool = False,
):
    #eng = matlab.engine.start_matlab()
    eng = matlab_engine
    _ = eng.addpath(eng.genpath("GenLouvain"), nargout=1)
    B_matlab = matlab.double(B.tolist())

    if random_state is not None:
        eng.eval(f"rng({int(random_state)});", nargout=0)
        if use_iterated:
            S, Q_raw = eng.iterated_genlouvain(B_matlab, [], [], 0, nargout=2)
        else:
            S, Q_raw = eng.genlouvain(B_matlab, nargout=2)

    S_py = np.array(S).reshape(-1)
    return S_py, float(Q_raw)

def hetero_genlouvain_communities(
    csv_path: str,
    omega_scale: float,
    matlab_engine: matlab.engine, 
    metric: str = "euclidean",
    node2vec_dim: int = 64,
    seed: int = 42,
):
    df = read_edges(csv_path)
    graphs = build_layer_graphs(df)

    embeddings = {}
    for layer, G in graphs.items():
        emb = compute_node2vec_embeddings(
            G,
            dimensions=node2vec_dim,
            walk_length=80,
            num_walks=10,
            p=1.0,
            q=1.0,
            window=10,
            workers=4,
            seed=seed,
        )
        embeddings[layer] = emb

    A_supra, global_index, layer_nodes, layer_adj_blocks, coupling_blocks = supra_adjacency_from_pi(
        graphs,
        embeddings,
        metric=metric,
        omega_scale=omega_scale,
    )

    B = build_B_from_blocks(layer_adj_blocks, coupling_blocks, layer_nodes, gamma=1.0)

    S_vec, Q_raw = run_hetero_genlouvain(
        B,
        matlab_engine=matlab_engine,
        random_state=seed,
        use_iterated=True,
    )
    rows = []
    for idx_global, (layer, node) in enumerate(global_index):
        rows.append({
            "layer": int(layer),
            "node_id": node,
            "community": int(S_vec[idx_global]),
        })

    output_csv = "output/test_communities.csv"
    part_df = pd.DataFrame(rows, columns=["node_id", "layer", "community"])
    part_df.to_csv(output_csv, index=False)
    return part_df, Q_raw, A_supra, B


#=====infomap + GW coupling 


def state_id(node: str, layer: str) -> str:
    # 选择一个不太可能出现在节点/层名里的分隔符
    return f"{node}|{layer}"


def build_infomap_with_gw(
    graphs: Dict[str, nx.Graph],
    embeddings: Dict[str, pd.DataFrame],
    layer_order: List[str],
    top_k: int = 5,
    min_weight: float = 1e-3,
    alpha: float = 2.0,
    emb_metric: str = "euclidean",
    intra_weight_from_attr: str = "weight"
) -> Tuple[Infomap, Dict[int, Tuple[str, str]]]:
    """
    构造扁平化“状态节点图”，节点是 (node, layer) 的组合映射为 int。
    返回 (Infomap实例, id2state)，其中 id2state[i] = (node, layer)
    """

    # ----------- 1️⃣ 建立全局状态节点映射 -----------
    state2id = {}
    id2state = {}
    next_id = 1

    # 遍历所有层的所有节点，创建唯一的整数ID
    for layer in layer_order:
        for n in graphs[layer].nodes():
            key = (str(n), str(layer))
            state2id[key] = next_id
            id2state[next_id] = key
            next_id += 1

    im = Infomap("--two-level --silent")

    # ----------- 2️⃣ 层内边 -----------
    for layer in layer_order:
        G = graphs[layer]
        for u, v, d in G.edges(data=True):
            w = float(d.get(intra_weight_from_attr, INTRA_EDGE_WEIGHT))
            uid = state2id[(str(u), str(layer))]
            vid = state2id[(str(v), str(layer))]
            im.addLink(int(uid), int(vid), float(w))

    # ----------- 3️⃣ 层间边 (GW 耦合) -----------
    for t in range(len(layer_order) - 1):
        la, lb = layer_order[t], layer_order[t + 1]
        emb_a, emb_b = embeddings[la].sort_index(), embeddings[lb].sort_index()
        pi, nodes_a, nodes_b, gw = gw_coupling_and_distance(emb_a, emb_b, metric=emb_metric)

        Omega = float(np.exp(-alpha * gw))  # 全局缩放
        row_sums = pi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = pi / row_sums

        na, nb = P.shape
        for ia in range(na):
            k_use = min(top_k, nb) if top_k > 0 else nb
            idx = np.argpartition(-P[ia, :], kth=k_use - 1)[:k_use]
            idx = idx[np.argsort(-P[ia, idx])]

            src_name = nodes_a[ia]
            src_id = state2id[(str(src_name), str(la))]

            for jb in idx:
                w = float(P[ia, jb]) * Omega
                if w < min_weight:
                    continue
                dst_name = nodes_b[jb]
                dst_id = state2id[(str(dst_name), str(lb))]
                im.addLink(int(src_id), int(dst_id), float(w))

    return im, id2state


# ============ RUN ============
def state_id(node: str, layer: str) -> str:
    return f"{node}|{layer}"


def build_infomap_with_gw(
    graphs: Dict[str, nx.Graph],
    embeddings: Dict[str, pd.DataFrame],
    layer_order: List[str],
    top_k: int = 5,
    min_weight: float = 1e-3,
    alpha: float = 2.0,
    emb_metric: str = "euclidean",
    intra_weight: int = 1
) -> Tuple[Infomap, Dict[int, Tuple[str, str]]]:
    state2id = {}
    id2state = {}
    next_id = 1

    for layer in layer_order:
        for n in graphs[layer].nodes():
            key = (str(n), str(layer))
            state2id[key] = next_id
            id2state[next_id] = key
            next_id += 1

    im = Infomap("--two-level --silent")

    for layer in layer_order:
        G = graphs[layer]
        for u, v, d in G.edges(data=True):
            w = float(1)
            uid = state2id[(str(u), str(layer))]
            vid = state2id[(str(v), str(layer))]
            im.addLink(int(uid), int(vid), float(w))

    for t in range(len(layer_order) - 1):
        la, lb = layer_order[t], layer_order[t + 1]
        emb_a, emb_b = embeddings[la].sort_index(), embeddings[lb].sort_index()
        pi, nodes_a, nodes_b, gw = gw_coupling_and_distance(emb_a, emb_b, metric=emb_metric)

        Omega = float(np.exp(-alpha * gw))
        row_sums = pi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = pi / row_sums

        na, nb = P.shape
        for ia in range(na):
            k_use = min(top_k, nb) if top_k > 0 else nb
            idx = np.argpartition(-P[ia, :], kth=k_use - 1)[:k_use]
            idx = idx[np.argsort(-P[ia, idx])]

            src_name = nodes_a[ia]
            src_id = state2id[(str(src_name), str(la))]

            for jb in idx:
                w = float(P[ia, jb]) * Omega
                if w < min_weight:
                    continue
                dst_name = nodes_b[jb]
                dst_id = state2id[(str(dst_name), str(lb))]
                im.addLink(int(src_id), int(dst_id), float(w))

    return im, id2state

def infomap_gw_coupling_communities(
        input_path,
        dimension=32,
        walk_length=80,
        num_walks=10,
        P=1.0,
        Q=1.0, 
        window=10,
        works=4, 
        seed=42,
        top_k=5,
        min_weight=1e-3,
        alpha=2.0,
        emb_metric="euclidean",

):
    df = read_edges(input_path)
    graphs = build_layer_graphs(df)
    layer_names = sorted(graphs.keys(), key=lambda x: x)  # 你也可以自定义顺序

    # 逐层嵌入
    embeddings: Dict[str, pd.DataFrame] = {}
    for layer in layer_names:
        embeddings[layer] = compute_node2vec_embeddings(
            graphs[layer],
            dimensions=dimension, walk_length=walk_length, num_walks=num_walks,
            p=P, q=Q, window=window, workers=works, seed=seed
        )

        
    im, id2state = build_infomap_with_gw(
        graphs, embeddings, layer_order=layer_names,
        top_k=top_k, min_weight=min_weight, alpha=alpha, emb_metric=emb_metric
    )

    im.run()
    partition = {n.node_id: n.module_id for n in im.tree if n.is_leaf}
    rows = []
    for nid, com in partition.items():
        if nid in id2state:
            node, layer = id2state[nid]
            rows.append((node, int(layer), com))

    assignments_df = pd.DataFrame(rows, columns=["node_id", "layer", "community"])  
    return assignments_df