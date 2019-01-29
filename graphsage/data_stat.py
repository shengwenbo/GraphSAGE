from graphsage.utils import load_data, split_date
import numpy as np
import json
import os


WEIGHTS = [1, 1, 8]

prefix = 'C:/reddit/reddit'
new_prefix = 'C:/reddit_simple/reddit'

G_data = json.load(open(prefix + "-G.json"))
if os.path.exists(prefix + "-feats.npy"):
    feats = np.load(prefix + "-feats.npy")
else:
    print("No features present.. Only identity features will be used.")
    feats = None
id_map = json.load(open(prefix + "-id_map.json"))
class_map = json.load(open(prefix + "-class_map.json"))

ids_train = [n['id'] for n in G_data['nodes'] if not n['val'] and not n['test']]
ids_else = [n['id'] for n in G_data['nodes'] if n['val'] or n['test']]
np.random.shuffle(ids_train)
np.random.shuffle(ids_else)

new_ids = ids_train[:len(ids_train)//100] + ids_else[:len(ids_else)//100]
new_G = {
    'directed': False,
    'graph': {},
    'nodes': [],
    'links': [],
    'multigraph': False
}
new_id_map = {}
idx = 0
for node in G_data['nodes']:
    if node['id'] in new_ids:
        new_G['nodes'].append(node)
        new_id_map[node['id']] = idx
        idx += 1

for source in range(len(new_ids)):
    for target in range(source):
        if np.random.rand(1) < 0.05:
            link = {}
            link["source"] = source
            link["target"] = target
            new_G["links"].append(link)
            link = {}
            link["source"] = target
            link["target"] = source
            new_G["links"].append(link)

new_class_map = {}
for id in new_ids:
    new_class_map[id] = class_map[id]

idxs = [id_map[id] for id in new_ids]
new_feats = feats[idxs]

json.dump(new_G, open(new_prefix + "-G.json", 'w'))
new_feats.dump(new_prefix + "-feats.npy")
json.dump(new_id_map, open(new_prefix+"-id_map.json", 'w'))
json.dump(new_class_map, open(new_prefix+"-class_map.json", 'w'))