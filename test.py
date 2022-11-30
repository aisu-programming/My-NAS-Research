import time
import torch

TEST_TURN = 64

t = torch.ones((64, 16, 32, 32))

start_time = time.time()
a = torch.ones((64, 0, 32, 32))
for _ in range(TEST_TURN):
    a = torch.concat([a, t], dim=1)
print(time.time()-start_time)

start_time = time.time()
b = torch.concat([t]*TEST_TURN, dim=1)
print(time.time()-start_time)

print(a.shape, b.shape)