import os
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_mutual_info_score

def compute_layerwise_ami_avg(
        gt_path: str, 
        pred) -> float:
    
    gt = pd.read_csv(gt_path)
    if isinstance(pred, str):
        pred = pd.read_csv(pred)
    elif not isinstance(pred, pd.DataFrame):
        raise TypeError("pred must be str or pandas.DataFrame.")

    for col in ["u", "v"]:
        if col in gt.columns:
            gt[col] = gt[col].astype(str)
    if "node_id" in pred.columns:
        pred["node_id"] = pred["node_id"].astype(str)

    gt_multilabels = defaultdict(lambda: defaultdict(list))

    for _, row in gt.iterrows():
        layer = row["layer"]
        gt_multilabels[layer][row["u"]].append(row["u_label"])
        gt_multilabels[layer][row["v"]].append(row["v_label"])

    gt_labels = defaultdict(dict)  # layer -> {node: label}
    for layer, node_lists in gt_multilabels.items():
        for node, labs in node_lists.items():
            cnt = Counter(labs)
            mode_label = cnt.most_common(1)[0][0]
            gt_labels[layer][node] = mode_label

    required_pred_cols = {"node_id","layer","community"}
    if not required_pred_cols.issubset(pred.columns):
        missing = required_pred_cols - set(pred.columns)
        raise ValueError(f"Predictions 缺少列: {missing}")

    pred_labels = defaultdict(dict)  # layer -> {node: community}
    for _, row in pred.iterrows():
        pred_labels[row["layer"]][row["node_id"]] = row["community"]

    amis = []
    for layer in sorted(set(gt_labels.keys()) & set(pred_labels.keys())):
        gt_nodes = set(gt_labels[layer].keys())
        pr_nodes = set(pred_labels[layer].keys())
        common = sorted(gt_nodes & pr_nodes)
        if not common:
            continue

        y_true = [gt_labels[layer][n] for n in common]
        y_pred = [pred_labels[layer][n] for n in common]

        ami = adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        amis.append(ami)
    print(amis)
    return float(pd.Series(amis).mean()) if len(amis) > 0 else float("nan")