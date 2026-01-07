import torch

def get_device(device_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_id)}")
    return torch.device("cpu")