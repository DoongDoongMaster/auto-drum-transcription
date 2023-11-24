import os
from glob import glob

DRUM = ['CC', 'HH', 'RC', 'ST', 'MT', 'SD', 'FT', 'KK', 'P1', 'P2']
BEAT = ['04', '08', '16']

dataset = glob('./dataset/*.m4a')
print(dataset)
drum = DRUM[8]
beat = BEAT[1]
start = 8

for d in dataset:
    new_name = f'./dataset/{drum}_{beat}_{start:04}.m4a'
    os.rename(d, new_name)
    start+=1