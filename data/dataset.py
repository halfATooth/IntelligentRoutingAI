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

def get_dataset(min=4, max=20):
    '''
    把读取到的数据聚类，得到类别标签
    '''
    data = read_data(min, max)
    # 使用cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    data = torch.tensor(data, dtype=torch.float32).to(device)
    # 归一化
    X = process.normalization(data)
    # 使用fcm聚类得到10个中心向量 和 每个特征点分别到10个中心点的距离
    _, Dist = process.iterate_cal(X.T)
    # 选择距离特征点最近的中心点的下标，作为特征点的类别标签
    y = torch.argmin(Dist, dim=0)
    print('dataset ready')
    return X.cpu(), y.cpu()
    
def visualize_clusters():
    '''
    使用TSNE把数据可视化
    '''
    X, y = get_dataset(min=9, max=10)
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

def save():
    X, y = get_dataset()
    tensor_dict = { 'feature': X, 'label': y }
    torch.save(tensor_dict, './data/tensor/n4n20.pt')

def load():
    tensor_dict = torch.load('./data/tensor/n4n20.pt')
    return tensor_dict['feature'], tensor_dict['label']
