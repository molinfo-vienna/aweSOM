import torch


def get_device() -> torch.device:
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_gpu_info() -> tuple[list[str], list[int]]:
    """
    Get information about the available GPU.

    Returns:
        tuple[list[str], list[int]]: (GPU name, GPU memory in GB)
    """
    gpu_names: list[str] = []
    gpu_memories: list[int] = []
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


def get_optimal_batch_size() -> int:
    """Automatically determine optimal batch size based on GPU memory."""
    if torch.cuda.is_available():
        device = get_device()
        gpu_memory = (
            torch.cuda.get_device_properties(device).total_memory / 1024**3
        )  # GB

        if gpu_memory >= 24:  # 24GB+ GPU (e.g., RTX 4090, A100)
            return 256
        elif gpu_memory >= 16:  # 16-24GB GPU (e.g., RTX 4080, V100)
            return 192
        elif gpu_memory >= 12:  # 12-16GB GPU (e.g., RTX 3080 Ti)
            return 128
        elif gpu_memory >= 8:  # 8-12GB GPU (e.g., RTX 3080, RTX 4070)
            return 96
        else:  # <8GB GPU
            return 64
    else:
        return 32  # CPU fallback
