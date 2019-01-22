from graphsage.utils import load_data
import numpy as np

WEIGHTS = [1, 1, 8]

print("Loading data...")
data = load_data("C:/reddit/reddit")
G = data[0]
classes = data[4]
print("Done loading data.")
train_cnt = 0
val_cnt = 0
test_cnt = 0

num_classes = 0

classes = {}

for id in G.node.keys():
    n = G.node[id]
    label = classes[id]

    if n['val']:
        val_cnt += 1
    elif n['test']:
        test_cnt += 1
    else:
        train_cnt += 1

    if label in classes.keys():
        classes[label].append(id)
    else:
        classes[label] = [id]

for label, ids in classes.items():


print("train: {}, val: {}, test:{}.\nclasses: {}.".format(train_cnt, val_cnt, test_cnt, num_classes))