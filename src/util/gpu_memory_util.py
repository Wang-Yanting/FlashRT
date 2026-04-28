import torch

def get_all_gpu_memory():
    peaks = {}
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)  # flush kernels
        peaks[d] = torch.cuda.max_memory_allocated(d)
    total_peak = sum(peaks.values())
    return total_peak/1024**3
