import os
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_mutual_info_score

def compute_layerwise_ami_avg(name: str, data_dir: str = "sync_data", output_dir: str = "output") -> float:

    gt_path = os.path.join(data_dir, f"{name}.csv")
    pred_path = os.path.join(output_dir, f"{name}_communities.csv")

    # 读取数据
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    # 统一节点ID为字符串，避免类型不一致导致的集合交集问题
    for col in ["u", "v"]:
        if col in gt.columns:
            gt[col] = gt[col].astype(str)
    if "node_id" in pred.columns:
        pred["node_id"] = pred["node_id"].astype(str)

    # -------- 构建逐层的真值节点标签 --------
    # 收集每个 (layer, node) 可能出现的多个标签（若图里多条边重复出现）
    gt_multilabels = defaultdict(lambda: defaultdict(list))
    required_gt_cols = {"u","v","layer","u_label","v_label"}
    if not required_gt_cols.issubset(gt.columns):
        missing = required_gt_cols - set(gt.columns)
        raise ValueError(f"Ground truth 缺少列: {missing}")

    for _, row in gt.iterrows():
        layer = row["layer"]
        gt_multilabels[layer][row["u"]].append(row["u_label"])
        gt_multilabels[layer][row["v"]].append(row["v_label"])

    # 将多次出现的标签用众数（mode）压缩为单标签；如并列众数，取第一个
    gt_labels = defaultdict(dict)  # layer -> {node: label}
    for layer, node_lists in gt_multilabels.items():
        for node, labs in node_lists.items():
            cnt = Counter(labs)
            mode_label = cnt.most_common(1)[0][0]
            gt_labels[layer][node] = mode_label

    # -------- 构建逐层的预测标签 --------
    required_pred_cols = {"node_id","layer","community"}
    if not required_pred_cols.issubset(pred.columns):
        missing = required_pred_cols - set(pred.columns)
        raise ValueError(f"Predictions 缺少列: {missing}")

    pred_labels = defaultdict(dict)  # layer -> {node: community}
    for _, row in pred.iterrows():
        pred_labels[row["layer"]][row["node_id"]] = row["community"]

    # -------- 逐层计算 AMI --------
    amis = []
    for layer in sorted(set(gt_labels.keys()) & set(pred_labels.keys())):
        gt_nodes = set(gt_labels[layer].keys())
        pr_nodes = set(pred_labels[layer].keys())
        common = sorted(gt_nodes & pr_nodes)
        if not common:
            continue

        y_true = [gt_labels[layer][n] for n in common]
        y_pred = [pred_labels[layer][n] for n in common]

        # sklearn 会把非数值标签自动作为分类标签处理
        ami = adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        amis.append(ami)
    print(amis)
    return float(pd.Series(amis).mean()) if len(amis) > 0 else float("nan")