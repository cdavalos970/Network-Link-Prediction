import json
import numpy as np
import pandas as pd
import csv

file = 'nodes.json'
matrix = np.zeros((4085,405), dtype = int)
print(matrix)
columns = []

with open(file) as f:
    data = json.load(f)
    f.close()

for author in data:
    id = author['id']
    for attribute, value in author.items():
        if attribute in columns:
            matrix[id, columns.index(attribute)] = int(value)
        else:
            columns.append(attribute)
            matrix[id, columns.index(attribute)] = int(value)

df = pd.DataFrame(matrix)
df.columns = columns
df.to_csv(r'authorMatrix.csv', index = False, header=True)