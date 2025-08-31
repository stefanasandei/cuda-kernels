import time

import torch


def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        exit()
    device = torch.device("cuda")

    # parameters
    size = 1000000
    num_trials = 10
    num_warmup_runs = 5

    x_gpu = torch.randn(size, device=device)
    y_gpu = torch.randn(size, device=device)

    # warm-up runs
    for _ in range(num_warmup_runs):
        result_gpu = add_vectors(x_gpu, y_gpu)
        torch.cuda.synchronize()

    times = []

    for i in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        result_gpu = add_vectors(x_gpu, y_gpu)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)

    mean_time = torch.tensor(times).mean().item()
    std_time = torch.tensor(times).std().item()

    print(f"Time: {mean_time:.3f} ± {std_time:.3f} ms")
