from pathlib import Path
import random

data_dir = Path('./dataset')
train_path = data_dir / "sentTrain.txt"
out_path = data_dir / "sentTrain_resample.txt"

counts = [0] * 35
with open(train_path, "r", encoding='utf8') as f:
    for line in f:
        content = line.split()
        label = int(content[-1])
        counts[label] += 1
relation_radio = sum(counts[1:]) / sum(counts)
oversample_radio = 1
undersample_radio = 3 * oversample_radio * sum(counts[1:]) / counts[0]
print(relation_radio, undersample_radio)

with open(train_path, "r", encoding='utf8') as f, open(out_path, "w", encoding='utf8') as g:
    for line in f:
        content = line.split()
        if content[-1] == "0":
            r = random.uniform(0, 1)
            if r <= undersample_radio:
                g.write(line)

        else:
            for i in range(oversample_radio):
                g.write(line)
