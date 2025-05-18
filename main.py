import posix_ipc
import mmap
import time
from model.model import GCN
import torch
from data.dataset import get_data_on_simulator

DATA_BLOCK_NAME = "/data_memory"
CONTROL_BLOCK_NAME = "/control_memory"

DATA_BLOCK_SIZE = 10240
CONTROL_BLOCK_SIZE = 32

duration = 610 # seconds
interval = 500 # ms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(4, 16, 7)
model.load_state_dict(torch.load('./model/gcn.pth'))
model = model.to(device)
model.eval()
meanstd = torch.load('./model/meanstd.pt')
mean = meanstd['mean'].to(device)
std = meanstd['std'].to(device)

receive = ''

def predict(data):
  print('receive: \n{0}'.format(data))
  global receive
  receive += data
  
  feature, topology, edge_map = get_data_on_simulator(data)
  feature = feature.to(device)
  feature = (feature - mean) / std
  topology = topology.to(device)
  
  out = model(feature, topology)
  pred = out.argmax(dim=1).cpu().tolist()
  print(pred)
  res = ''
  for i in range(len(edge_map)):
    res = res + f'{edge_map[i][0]} {edge_map[i][1]} {pred[i] + 1}/'
  res = res.strip('/')
  return res

def getPaddedLength(s):
  length = len(s)
  length_str = str(length).zfill(8)
  return length_str

def read(memMap):
  msg = memMap.read().decode().rstrip('\x00')
  memMap.seek(0)
  return msg

def write(memMap, data):
  memMap.write(data.encode())
  memMap.seek(0)

try:
  # open shared memory of data block and control block
  dataShm = posix_ipc.SharedMemory(DATA_BLOCK_NAME, posix_ipc.O_CREAT, size=DATA_BLOCK_SIZE)
  dataMemMap = mmap.mmap(dataShm.fd, dataShm.size)
  controlShm = posix_ipc.SharedMemory(CONTROL_BLOCK_NAME, posix_ipc.O_CREAT, size=CONTROL_BLOCK_SIZE)
  controlMemMap = mmap.mmap(controlShm.fd, controlShm.size)
  print('start')
  # round robin
  for _ in range(int(duration * 1000 / interval)):
    controlMessage = read(controlMemMap)
    if controlMessage[:2] == 'ai': # data to ai ready
      dataSize = int(controlMessage[3:])
      # get link data and topology data of network
      data = read(dataMemMap)
      data = data[:dataSize]
      # call ai program and get weight data of all links
      data = predict(data)
      # transfer to ns3
      write(dataMemMap, data)
      write(controlMemMap, 'ns/{0}'.format(getPaddedLength(data)))
    time.sleep(interval / 1000)
        
  # 释放共享内存在ns3端执行
    
except posix_ipc.ExistentialError as e:
    print(f"共享内存操作出错: {e}")

with open('./data/evalafter', 'w') as file:
    file.writelines(receive)
# print(receive)
# arr = receive.strip().split('\n')
# delay = []
# drop = []
# for str in arr:
#   d = str.split(' ')
#   delay.append(d[2])
#   drop.append(d[4])
  
# delay = torch.mean(torch.tensor(delay))
# drop = torch.mean(torch.tensor(drop))
# print(delay)
# print(drop)