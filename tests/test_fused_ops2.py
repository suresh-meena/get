import torch
import time

def main():
    B, M, F = 32, 2048, 64
    Q = torch.randn(B, M, F, device="cuda")
    Ku = torch.randn(B, M, F, device="cuda")
    Kv = torch.randn(B, M, F, device="cuda")
    T = torch.randn(M, F, device="cuda")

    def torch_einsum():
        return torch.einsum("bmf,bmf->bm", Q, Ku * Kv + T)

    def torch_sum():
        return torch.sum(Q * (Ku * Kv + T), dim=-1)

    # warmup
    for _ in range(10):
        torch_einsum()
        torch_sum()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        torch_einsum()
    torch.cuda.synchronize()
    t1 = time.time()

    for _ in range(100):
        torch_sum()
    torch.cuda.synchronize()
    t2 = time.time()

    print(f"Einsum: {t1 - t0:.4f}s")
    print(f"Sum: {t2 - t1:.4f}s")


if __name__ == "__main__":
    main()
