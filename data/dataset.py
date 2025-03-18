import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import torch
import process
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

def generate_random_connected_graphs(n, maxEdges, m):
    """
    生成m个满足以下条件的无向图:
        1、整个图是连通的且没有自环
        2、图的顶点数为n
        3、图的边数小于等于maxEdges
    """
    if(maxEdges > n*(n-1)/2):
        maxEdges = n*(n-1)/2
    result = []
    while len(result) < m:
        num_edges = random.randint(n - 1, maxEdges)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        all_possible_edges = list(nx.non_edges(G))
        edges = random.sample(all_possible_edges, num_edges)
        G.add_edges_from(edges)
        if nx.is_connected(G):
            result.append(G)
    return result

def visualize_graphs(graphs, output_path='output.png'):
    """
    可视化多个无向图
    :param graphs: 包含多个networkx图的列表
    """
    plt.figure(figsize=(15, 5))
    for i, G in enumerate(graphs, 1):
        plt.subplot(1, len(graphs), i)
        nx.draw(
            G,
            with_labels=True,
            node_color='skyblue',
            edge_color='gray',
            pos=nx.spring_layout(G, seed=42),
            node_size=500  # 节点大小
        )
        plt.title(f"Graph {i}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def write_graphs_to_file(graphs, dir):
    '''
    将拓扑数据写入文件
    '''
    for i, graph in enumerate(graphs, 1):
        filename = f'{dir}/{i}/topology'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            for edge in list(graph.edges()):
                file.write(f'{edge[0]} {edge[1]}\n')

def generate_topology_data(minNodes:int=4, maxNodes:int=30, graphNum:int=20, dataDir:str='./data/net'):
    '''
    生成拓扑数据
    '''
    for n in range(minNodes, maxNodes+1):
        graphs = generate_random_connected_graphs(n, int(n*1.6), graphNum)
        dir = f'{dataDir}/nodes_num_{n}'
        write_graphs_to_file(graphs, dir)
        
def read_data(min=4, max=20):
    '''
    读取ns-3生成的网络状态数据
    '''
    data = []
    try:
        for node in range(min, max+1):
            for topo in range(1, 20):
                with open(f'./data/net/nodes_num_{node}/{topo}/features', 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line in lines:
                        if len(line) > 2:
                            d = [float(num) for num in line.split(' ')[2:]]
                            data.append(d)
    except FileNotFoundError:
        print("文件未找到")
    print('read data complete')
    return data

def get_dataset(min=4, max=20, m=2, default_data='./data/tensor/raw.pt'):
    '''
    把读取到的数据聚类，得到类别标签
    '''
    # 使用cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    try:
        X = torch.load(default_data).to(device)
    except FileNotFoundError:
        data = read_data(min, max)
        data = torch.tensor(data, dtype=torch.float32).to(device)
        # 归一化
        X = process.normalization(data)
        torch.save(X, default_data)
    
    try:
        center_weights = torch.load('./data/tensor/center-weights.pt')
        centers = center_weights['centers'].to(device)
        weights = center_weights['weights'].to(device)
        dist = process.cal_distance_matrix(centers.T, X.T)
        # top_values, top_indices = torch.topk(-dist, k=3, dim=0)
        y = torch.argmin(dist, dim=0)
        y = weights[y]
        
    except FileNotFoundError:
        print('未找到人工设定的权值，进行fcm聚类')
        # 使用fcm聚类得到10个中心向量 和 每个特征点分别到10个中心点的距离
        centers, Dist = process.iterate_cal(X.T, m=m)
        # 选择距离特征点最近的中心点的下标，作为特征点的类别标签
        # 这样计算得到的类别并无实际意义，需要进一步人工筛选得到类别-权值的对应关系
        y = torch.argmin(Dist, dim=0)
    print('dataset ready')
    return X.cpu(), y.cpu(), centers.cpu()

def visualize_clusters():
    '''
    使用TSNE把数据可视化
    '''
    X, y, _ = get_dataset(min=9, max=20)
    X = X.numpy()
    y = y.numpy()
    # 创建TSNE对象，将数据降维到二维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    # 可视化降维后的数据
    plt.figure(figsize=(10, 8))
    # 根据不同的标签绘制不同颜色的点
    for i in range(len(np.unique(y))):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
    plt.title('T-SNE Visualization of Dataset')
    # plt.legend()
    plt.savefig('tsne-2D2.png')

def save(filename):
    X, y, c = get_dataset()
    tensor_dict = { 'feature': X, 'label': y, 'center': c }
    torch.save(tensor_dict, f'./data/tensor/{filename}.pt')

def load(path):
    tensor_dict = torch.load(path)
    return tensor_dict['feature'], tensor_dict['label'], tensor_dict['center']

def set_weight():
    _, _, c = get_dataset(m=2.5)
    c = c.T
    for i in range(c.size(0)):
        print(f'{c[i,0]}#{c[i,1]}#{c[i,2]}#{c[i,3]}')

def set_centers():
    centers = torch.tensor([
        [-0.585467696, 0.008220225, -0.551420867, -0.765355527],
        [2.087510824, -0.013327285, 0.391151488, -0.274538726],
        [0.144209802, 0.625683963, 0.352233231, 0.496681154],
        [0.137531653, -0.645043314, 0.362915069, 0.498750418],
        [0.346270621, -0.010934002, 0.359801352, 0.353592128],
        [-0.606748998, -1.089063764, -0.56386143, -0.785086215],
        [-0.084792867, 0.009439957, 0.355262339, 1.188973904],
        [-0.604935825, 1.101434588, -0.563349009, -0.782497406]
    ])
    weights = torch.tensor([9, 5, 2, 6, 5, 10, 4, 8])
    tensor_dict = { 'centers': centers, 'weights': weights }
    torch.save(tensor_dict, f'./data/tensor/center-weights.pt')
