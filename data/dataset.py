import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import json
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

def get_transformed_graph(node, topo):
    edges = []
    node_map = {}
    new_edges = []
    with open(f'./data/net/nodes_num_{node}/{topo}/topology', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            u, v = map(int, line.split())
            node_map[(u, v)] = i
            edges.append((u, v))
        # 遍历每一对边
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                edge1 = edges[i]
                edge2 = edges[j]
                # 检查两条边是否有公共顶点
                if edge1[0] in edge2 or edge1[1] in edge2:
                    new_edges.append((i, j))
    return node_map, new_edges

def transform_raw_data(min, max):
    '''
    把原始数据预处理：
    1. 拓扑边点互换
    2. 转json格式:
        第一维: node顶点个数
        第二维: topo拓扑结构编号
        第三维: content = {'topo': new_edges, 'features': feature_arr, 'len': ct}
        feature_arr: 二维数组，顶点编号*特征值
    '''
    try:
        node_content = []
        for node in range(min, max+1):
            topo_content = []
            for topo in range(1, 20):
                node_map, new_edges = get_transformed_graph(node, topo)
                # ct记录当前的拓扑有多少条特征
                ct = 0
                feature_arr = []
                with open(f'./data/net/nodes_num_{node}/{topo}/features', 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    graph_features = {}
                    for line in lines:
                        line = line.rstrip('\n')
                        if line=='{':
                            continue
                        elif line=='}':
                            ct = ct + len(graph_features)
                            arr = [None] * len(graph_features)
                            for k, v in graph_features.items():
                                arr[k] = v
                            feature_arr.extend(arr)
                            graph_features = {}
                        else:
                            d = [num for num in line.split(' ')]
                            # transformed node index
                            if int(d[0]) > int(d[1]):
                                index = node_map[(int(d[1]), int(d[0]))]
                            else:
                                index = node_map[(int(d[0]), int(d[1]))]
                            delay_avg = float(d[2])
                            bandwith = float(d[3])
                            droprate_avg = float(d[4])
                            throughoutput = float(d[5])
                            if index in graph_features:
                                graph_features[index][0] = (graph_features[index][0] + delay_avg) / 2
                                graph_features[index][2] = (graph_features[index][2] + droprate_avg) / 2
                                graph_features[index][3] = graph_features[index][3] + throughoutput
                            else:
                                graph_features[index] = [delay_avg, bandwith, droprate_avg, throughoutput]
                content = {'topo': new_edges, 'features': feature_arr, 'len': ct}
                topo_content.append(content)
            node_content.append(topo_content)
        jsonstr = json.dumps(node_content)
        with open('./data/transformed_data/raw.json', 'w') as file:
            file.write(jsonstr)
    except FileNotFoundError:
        print("文件未找到")

def read_data(file_path='./data/transformed_data/raw.json'):
    '''
    读取ns-3生成的网络状态数据
    '''
    data = []
    if not os.path.isfile(file_path):
        transform_raw_data(4, 20)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw = json.load(file)
            for node_content in raw:
                for topo_content in node_content:
                    feature_arr = topo_content['features']
                    data.extend(feature_arr)
    except Exception as e:
        print(f"err: {e}")
    return data

def get_dataset(m=2, default_data='./data/tensor/raw.pt'):
    '''
    把读取到的数据聚类，得到类别标签
    '''
    # 使用cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if os.path.isfile(default_data):
        X = torch.load(default_data).to(device)
    else:
        data = read_data()
        data = torch.tensor(data, dtype=torch.float32).to(device)
        # 归一化
        X = process.normalization(data)
        torch.save(X, default_data)
    
    center_vector_path = './data/tensor/center-weights.pt'
    if os.path.isfile(center_vector_path):
        center_weights = torch.load(center_vector_path)
        centers = center_weights['centers'].to(device)
        weights = center_weights['weights'].to(device)
        dist = process.cal_distance_matrix(centers.T, X.T)
        # top_values, top_indices = torch.topk(-dist, k=3, dim=0)
        y = torch.argmin(dist, dim=0)
        y = weights[y]
    else:
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
    topology = []
    with open('./data/transformed_data/raw.json', 'r', encoding='utf-8') as file:
        raw = json.load(file)
        for node_content in raw:
            for topo_content in node_content:
                # 根据拓扑生成稀疏矩阵
                topo = torch.tensor(topo_content['topo'], dtype=torch.long).T
                row_indices = topo[0, :]
                col_indices = topo[1, :]
                size = col_indices.size()[0]
                symmetric_row_indices = torch.cat((row_indices, col_indices))
                symmetric_col_indices = torch.cat((col_indices, row_indices))
                values = torch.full((size*2,), 1.0, dtype=torch.float)
                adj_matrix = torch.sparse_coo_tensor(
                    torch.stack((symmetric_row_indices, symmetric_col_indices), dim=0),
                    values,
                    (size, size))
                # 对应拓扑id
                num = topo_content['len']
                topology.append((adj_matrix, num))
    X, y, _ = get_dataset()
    flag = 0
    res = []
    for i, (adj_matrix, num) in enumerate(topology):
        x_chunks = torch.chunk(X[flag: flag+num], 100, dim=0)
        y_chunks = torch.chunk(y[flag: flag+num], 100, dim=0)
        for xy in tuple(zip(x_chunks, y_chunks)):
            data = { 'feature': xy[0], 'label': xy[1], 'topo': adj_matrix }
            res.append(data)
        flag += num
    torch.save(res, f'./data/tensor/{filename}.pt')

def load(path):
    tensor_dict = torch.load(path)
    return tensor_dict['feature'], tensor_dict['label'], tensor_dict['center']

def plot_radar_charts(data, dimension_labels):
    num_vars = len(data[0])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    num_plots = len(data)
    rows = (num_plots + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for i, row in enumerate(data):
        values = row.tolist()
        values += values[:1]
        ax = axes[i]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_labels)
        ax.set_rlabel_position(0)

        labels = [-1, -0.5, 0, 0.5, 1]
        ax.set_yticks(labels)
        for tick in ax.get_yticklabels():
            tick.set_color("grey")
            tick.set_size(7)
        ax.set_ylim(-1, 1)

    # 隐藏多余的子图
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
    # plt.show()
    plt.savefig('./data/set-weight.png')


def set_weight():
    _, _, c = get_dataset(m=2.5)
    c = c.T
    torch.save(c, './data/tensor/center-vectors.pt')
    c[:, [0, 2, 3]] *= -1
    c[:, [0, 1, 2, 3]] = c[:, [1, 0, 3, 2]]
    dimension_labels = ['bandwith', 'speed', '-throughput', 'PDR']
    plot_radar_charts(c, dimension_labels)

def set_centers():
    centers = torch.load('./data/tensor/center-vectors.pt')
    centers = centers[[0, 1, 2, 4, 6, 7, 8], :]
    centers[:, [0, 1, 2, 3]] = centers[:, [1, 0, 3, 2]]
    centers[:, [0, 2, 3]] *= -1
    weights = torch.tensor([6, 7, 4, 1, 3, 2, 5])
    tensor_dict = { 'centers': centers, 'weights': weights }
    torch.save(tensor_dict, f'./data/tensor/center-weights.pt')
