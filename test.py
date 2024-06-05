import torch
import torch.cuda
# Set the dimensions
B = 2
NH = 4
T = 3
HS = 5

# Create random example matrices q and k
q = torch.randn((B, NH, T, HS), dtype=torch.float32)
k = torch.randn((B, NH, T, HS), dtype=torch.float32)

preatt_torch = q @ k.transpose(-2, -1)

q_cuda = q.cuda()
k_cuda = k.cuda()

preatt_cuda = torch.empty((B, NH, T, T), dtype=torch.float32, device='cuda')

handle = torch.cuda.current_blas_handle()

alpha = 1.0
beta = 0.0

torch.cuda.cublas.sgemmStridedBatched(handle, 'n', 't', T, T, HS, alpha, q_cuda.data_ptr(), HS, T * HS, k_cuda.data_ptr(), HS, T * HS, beta, preatt_cuda.data_ptr(), T, T * T, B * NH)

preatt_cuda_cpu = preatt_cuda.cpu()
if torch.allclose(preatt_torch, preatt_cuda_cpu, rtol=1e-4, atol=1e-4):
    print("The results match!")
else:
    print("The results do not match.")