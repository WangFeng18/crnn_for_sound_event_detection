import torch
from collections import OrderedDict as od

name = 'NETA_sl5.0.pth'
tname = 'NETA_sl5.0_X.pth'
b = od()
a = torch.load(name)
for k,v in a.items():
    if k.startswith('mo'):
        b[k[7:]] = v
torch.save(b, tname)

