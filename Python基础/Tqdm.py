# 方法1：
import time
from tqdm import tqdm

# num = 0
# for i in tqdm(range(101)):
#     num += i
#     time.sleep(0.01)

# # print(num)
# # 方法2：
# import time
# from tqdm import trange
#
# for i in trange(100):
#     time.sleep(0.01)
#

import time
from tqdm import tqdm

# 一共200个，每次更新10，一共更新20次
with tqdm(total=200) as pbar:
  for i in range(20):
    pbar.update(10)
    time.sleep(0.1)

#方法2：
pbar = tqdm(total=200)
for i in range(20):
    pbar.update(10)
    time.sleep(0.1)
# close() 不要也没出问题？
pbar.close()