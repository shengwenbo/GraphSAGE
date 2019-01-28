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

ids = [n['id'] for n in G_data['nodes']]
np.random.shuffle(ids)

new_ids = ids[:len(ids)//100]
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

for link in G_data['links']:
    if link['source'] in new_ids and link['target'] in new_ids:
        new_G['links'].append(link)

new_class_map = {}
for id in new_ids:
    new_class_map[id] = class_map[id]

idxs = [id_map[id] for id in new_ids]
new_feats = feats[idxs]

json.dump(new_G, open(new_prefix + "-G.json", 'w'))
feats.dump(new_prefix + "-feats.npy")
json.dump(new_id_map, open(new_prefix+"-id_map.json", 'w'))
json.dump(new_class_map, open(new_prefix+"-class_map.json", 'w'))