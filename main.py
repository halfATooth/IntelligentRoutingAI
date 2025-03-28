import posix_ipc
import mmap
import time

DATA_BLOCK_NAME = "/data_memory"
CONTROL_BLOCK_NAME = "/control_memory"

DATA_BLOCK_SIZE = 1024
CONTROL_BLOCK_SIZE = 1024

duration = 100 # seconds
interval = 500 # ms

def predict(data):
  print('receive: \n{0}'.format(data))
  return '0 1 1/0 2 1/1 3 1/2 3 1'

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
