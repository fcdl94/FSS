import torch
import sys

print("Elaborating checkpoint in: " + sys.argv[1])
ckpt = torch.load(sys.argv[1])

print(ckpt['epoch'])

for k,v in ckpt['trainer_state']['regularizer']['score'].items():
    if torch.isnan(v.sum()) or torch.isinf(v.sum()):
        print("score " + k)

if "RW" in sys.argv[1]:
    for k,v in ckpt['trainer_state']['regularizer']['fisher'].items():
        if torch.isnan(v.sum()) or torch.isinf(v.sum()):
            print("fisher " + k)
