from typing import List, Tuple

import torch


def get_device() -> torch.device:
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_gpu_info() -> Tuple[List[str], List[int]]:
    """
    Get information about the available GPU.

    Returns:
        Tuple[List[str], List[int]]: (GPU name, GPU memory in GB)
    """
    gpu_names: List[str] = []
    gpu_memories: List[int] = []
    for i in range(torch.cuda.device_count()):
        gpu_names.append(torch.cuda.get_device_name(i))
        gpu_memories.append(
            int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
        )
    return gpu_names, gpu_memories


def print_device_info() -> None:
    """Print basic device information."""
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        gpu_info = get_gpu_info()
        for gpu_name, gpu_memory in zip(gpu_info[0], gpu_info[1]):
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory} GB")
            print(f"CUDA Version: {torch.version.cuda}")
