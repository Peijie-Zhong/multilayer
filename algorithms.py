import pandas as pd
from infomap import Infomap
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from scipy import sparse
import matlab.engine
import matlab


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
    output_df.to_csv(output_csv_path, index=False)
    print(f"{input_csv_path} complete.")

# ===================GenLouvain===================



def build_multilayer_adjacency(
    df: pd.DataFrame,
    directed: bool = False,
) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
    """
    Build per-layer adjacency matrices with a consistent node ordering across layers.

    Returns:
        A_list: list of (N x N) numpy arrays for each layer (in sorted layer order)
        layers: ordered list of unique layer identifiers (as strings)
        node_index: dict mapping original node_id -> index [0..N-1]
    """
    # Tolerate case-insensitive column names
    lower_map = {c.lower(): c for c in df.columns}
    for col in ("u", "v", "layer"):
        if col not in lower_map:
            raise ValueError(
                f"Missing required column '{col}'. Found columns: {list(df.columns)}"
            )
    df = df.rename(columns={lower_map["u"]: "u", lower_map["v"]: "v", lower_map["layer"]: "layer"})

    df = df.dropna(subset=["u", "v", "layer"]).copy()
    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)

    # Sort layers numerically if possible, otherwise lexicographically
    try:
        df["_layer_num"] = pd.to_numeric(df["layer"], errors="raise")
        layers_sorted_num = sorted(df["_layer_num"].unique().tolist())
        layer_text_for_num = {
            ln: str(df.loc[df["_layer_num"] == ln, "layer"].iloc[0]) for ln in layers_sorted_num
        }
        layers_out = [layer_text_for_num[ln] for ln in layers_sorted_num]
        layer_key = "_layer_num"
    except Exception:
        layers_out = sorted(df["layer"].astype(str).unique().tolist())
        layer_key = "layer"

    # Node universe across all layers
    nodes = sorted(set(df["u"]).union(set(df["v"])), key=lambda x: (len(x), x))
    node_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    A_list: List[np.ndarray] = []
    for L in layers_out:
        if layer_key == "_layer_num":
            # get numeric key for this textual layer
            numeric = [k for k, v in layer_text_for_num.items() if v == L][0]
            sub = df[df["_layer_num"] == numeric]
        else:
            sub = df[df["layer"].astype(str) == L]

        if sub.empty:
            A_list.append(np.zeros((N, N), dtype=float))
            continue

        ui = sub["u"].map(node_index).to_numpy()
        vi = sub["v"].map(node_index).to_numpy()

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
    use_iterated: bool = True,
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
    if use_iterated and has_iter:
        eng.eval('[S, Q] = iterated_genlouvain(B, [], [], 0);', nargout=0)
    else:
        eng.eval('[S, Q] = genlouvain(B);', nargout=0)

    S = eng.workspace['S']

    # Convert S (length N*T vector in MATLAB) to N x T numpy array
    S_py = np.array(S).ravel(order="F")
    if S_py.size != N * T:
        S_py = np.array(list(np.array(S).flatten()), dtype=float).reshape((N * T,), order="F")
    S_mat = S_py.reshape((N, T), order="F")
    return S_mat


def genlouvain_communities(
    input_csv: str,
    output_csv: str,
    genlouvain_path: str,
    omega: float = 1.0,
    gamma: float | List[float] | np.ndarray = 1.0,
    directed: bool = False,
    random_state: int | None = None,
    use_iterated: bool = True,
) -> None:
    """
    End-to-end pipeline:
      1) read CSV and build multilayer adjacency
      2) call MATLAB GenLouvain
      3) write CSV of assignments: node_id, layer, community
    """
    df = pd.read_csv(input_csv)
    A_list, layers, node_index = build_multilayer_adjacency(df, directed=directed)

    S = call_genlouvain(
        A_list=A_list,
        genlouvain_path=genlouvain_path,
        omega=omega,
        gamma=gamma,
        use_iterated=use_iterated,
        random_state=random_state,
    )

    # Prepare output rows
    nodes_by_index = {i: n for n, i in node_index.items()}
    N, T = S.shape
    rows = []
    for t_idx, layer in enumerate(layers):
        for i in range(N):
            rows.append({
                "node_id": nodes_by_index[i],
                "layer": layer,
                "community": int(S[i, t_idx]),
            })

    out_df = pd.DataFrame(rows, columns=["node_id", "layer", "community"])
    # Stable sort by node, then layer (numeric if possible)
    try:
        out_df["_layer_sort"] = pd.to_numeric(out_df["layer"], errors="coerce")
        out_df = out_df.sort_values(by=["node_id", "_layer_sort", "layer"]).drop(columns=["_layer_sort"]) 
    except Exception:
        out_df = out_df.sort_values(by=["node_id", "layer"]) 

    out_df.to_csv(output_csv, index=False)


'''
run_genlouvain_multilayer(
    input_csv="sync_data/test.csv",
    output_csv="test_communities.csv",
    genlouvain_path="GenLouvain",
    omega=1.0,
    gamma=1.0,
    directed=False,
    random_state=None,
    use_iterated=True,
)
'''

