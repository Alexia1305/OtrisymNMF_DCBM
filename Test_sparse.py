from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import numpy as np
from scipy import sparse
import time
n=10000
A=np.random.rand(n,n)
M=sparse.random(n,n,density=0.0001,format='csr', dtype=np.float64, data_rvs=np.random.rand)
print("csr")
start_time = time.time()
M*A
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
M.T*A
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
A*M
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
A*M.T
end_time = time.time()
print(end_time - start_time)

print("csc")
M=M.tocsc()
start_time = time.time()
M*A
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
M.T*A
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
A*M
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
A*M.T
end_time = time.time()
print(end_time - start_time)