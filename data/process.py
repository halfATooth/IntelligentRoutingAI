import torch

def cal_distance_matrix(V, X):
    # 扩展维度以实现向量化计算
    X_expanded = X.unsqueeze(1)  # 形状变为 (m, 1, n)
    V_expanded = V.unsqueeze(2)  # 形状变为 (m, p, 1)
    # 使用广播机制，计算差值的平方和
    diff = X_expanded - V_expanded
    squared_diff = diff ** 2
    sum_squared_diff = torch.sum(squared_diff, dim=0)
    # 计算欧氏距离
    D = torch.sqrt(sum_squared_diff)
    return torch.nan_to_num(D)
     

# 迭代优化的目标函数
def J_FCM(Um, V, X):
    # Um_ij: 第j个样本对于第i个类别的隶属度的m次方
    Dist = cal_distance_matrix(V, X)
    return torch.sum(Um * Dist)

# def cal_U(D, m):
#     exp = -m/(m-1)
#     class_num = D.size(0)
#     data_num = D.size(1)
#     U = torch.zeros(class_num, data_num)
#     for i in range(class_num):
#         for j in range(data_num):
#             t = torch.zeros(class_num)
#             for k in range(class_num):
#                 t[k] = torch.pow(D[i, j] / D[k, j], 2)
#             U[i, j] = torch.pow(torch.sum(t), exp)
#     return U
def cal_Um(D, m):
    exp = -m / (m - 1)
    class_num = D.size(0)
    # data_num = D.size(1)
    # 扩展维度以实现向量化计算
    D_expanded = D.unsqueeze(0)  # 形状变为 (1, class_num, data_num)
    D_repeated = D.unsqueeze(1).repeat(1, class_num, 1)  # 形状变为 (class_num, class_num, data_num)
    # 计算中间结果 t
    t = D_expanded / D_repeated
    t = torch.nan_to_num(t)
    t = torch.pow(t, 2)
    # 对 t 沿着第 0 维求和
    sum_t = torch.sum(t, dim=0)
    # 计算最终结果 U
    return torch.pow(sum_t, exp)
     

# def cal_V(Um, X):
#     class_num = Um.size(0)
#     dim = X.size(0)
#     data_num = X.size(1)
#     V = torch.zeros(dim, class_num)
#     for i in range(class_num):
#         a = torch.zeros(dim)
#         b = 0.0
#         for j in range(data_num):
#             a = a + Um[i,j]*X[:,j]
#             b = b + Um[i,j]
#         V[:,i] = a / b
#     return V
def cal_V(Um, X):
    # 计算分子
    a = torch.matmul(Um, X.T)
    # 计算分母
    b = torch.sum(Um, dim=1, keepdim=True)
    return a.T / b.T

def iterate_cal(X, class_num=10, eps=1e-4, max_round=50, m=2):
    # m 是一个加权指数，随着 m 的增大，聚类的模糊性增大, m>1
    data_num = X.size(1)
    # # 随机选取 class_num 个列索引
    # selected_indices = torch.randperm(data_num)[:class_num]
    # # 根据选取的列索引从矩阵 X 中获取相应的列向量，组成新的矩阵 V
    # V = X[:, selected_indices]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    V = torch.randn(X.size(0), class_num).to(device)
    D = cal_distance_matrix(V, X)
    Um = cal_Um(D, m)
    delta = 1 # 每轮迭代后目标函数值的变化量
    last_res = J_FCM(Um, V, X)
    for _ in range(max_round):
        if abs(delta) < eps:
            break
        V = cal_V(Um, X)
        D = cal_distance_matrix(V, X)
        Um = cal_Um(D, m)
        fcm = J_FCM(Um, V, X)
        delta = last_res - fcm
        last_res = fcm
    return V, D

def normalization(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data = (data - mean) / std
    return data
