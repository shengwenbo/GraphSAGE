from graphsage.utils import load_data, split_date
import numpy as np
import json
import os
from networkx.readwrite import json_graph

WEIGHTS = [1, 1, 8]

prefix = 'C:/reddit/reddit'
new_prefix = 'C:/reddit_new/reddit'

G_data = json.load(open(prefix + "-G.json"))
if os.path.exists(prefix + "-feats.npy"):
    feats = np.load(prefix + "-feats.npy")
else:
    print("No features present.. Only identity features will be used.")
    feats = None
id_map = json.load(open(prefix + "-id_map.json"))
class_map = json.load(open(prefix + "-class_map.json"))

G = json_graph.node_link_graph(G_data)
G = split_date(G, class_map, [1, 1, 98])
new_G = {
    'directed': G_data['directed'],
    'graph': G_data['graph'],
    'nodes': [],
    'links': G_data['links'],
    'multigraph': G_data['multigraph']
}

for id,node in G.node.items():
    new_G["nodes"].append({
        'test': node['test'],
        'val': node['val'],
        'id': id
    })

new_feats = feats
new_id_map = id_map
new_class_map = class_map

json.dump(new_G, open(new_prefix + "-G.json", 'w'))
new_feats.dump(new_prefix + "-feats.npy")
json.dump(new_id_map, open(new_prefix+"-id_map.json", 'w'))
json.dump(new_class_map, open(new_prefix+"-class_map.json", 'w'))