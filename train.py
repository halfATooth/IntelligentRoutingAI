import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model.model import GCN


# 初始化模型、优化器和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据集
# dataset = torch.load('./data/tensor/data1.pt')
features = torch.load('./data/tensor/features.pt')
features = [x.to(device) for x in features]
labels = torch.load('./data/tensor/labels.pt')
labels = [y.to(device) for y in labels]
topology = torch.load('./data/tensor/topology.pt')
topology = [t.to(device) for t in topology]



model = GCN(4, 16, 7).to(device)
# data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)


# 训练模型
def train(X, y, topo):
    model.train()
    optimizer.zero_grad()
    out = model(X, topo)
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试模型
def test(X, y, topo):
    model.eval()
    out = model(X, topo)
    pred = out.argmax(dim=1)
    # print(y.cpu())
    # print(pred.cpu())
    test_correct = pred == y
    test_acc = int(test_correct.sum()) / int(test_correct.size()[0])
    return test_acc

ct = 0
# 训练循环
try:
    for epoch in range(1):
        for i in range(len(features)-100):
            # X, y, topo = data
            X = features[i]
            y = labels[i]
            topo = topology[i]
            loss = train(X, y, topo)
            if (ct + 1) % 100 == 0:
                test_acc = test(X, y, topo)
                print(f'Epoch: {(ct + 1)//100}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
            ct += 1
    model = model.cpu()
    torch.save(model.state_dict(), './model/gcn.pth')
except Exception as e:
    print(f'err: {e}')