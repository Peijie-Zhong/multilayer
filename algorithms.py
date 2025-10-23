import pandas as pd
from infomap import Infomap

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


