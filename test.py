import torch
from model.model import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(4, 16, 7)
model.load_state_dict(torch.load('./model/gcn.pth'))
model = model.to(device)
model.eval()

features = torch.load('./data/tensor/features.pt')
features = [x.to(device) for x in features]
labels = torch.load('./data/tensor/labels.pt')
labels = [y.to(device) for y in labels]
topology = torch.load('./data/tensor/topology.pt')
topology = [t.to(device) for t in topology]

print(len(features))

for i in range(1, 100):
  x = features[len(features)-i]
  t = topology[len(features)-i]
  out = model(x, t)
  pred = out.argmax(dim=1).cpu().tolist()
  print(pred)