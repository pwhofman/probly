from __future__ import annotations

import os

import torch


# Load data from a .pt file
def load_distributions(load_path: str, device: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """Load a tensor dictionary from a .pt file.

    Parameters:
        load_path: Path to load from.
        device:    Target device (e.g. 'cpu', 'cuda:0').
                   None keeps the original device.
        verbose:   Whether to print detailed information.

    Returns:
        A tensor dictionary.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File not found: {load_path}")

    # Load dictionary
    tensor_dict = torch.load(load_path, map_location=device)

    if verbose:
        print(f"Tensor dict has been loaded from {load_path}")
        print("Loaded contents overview:")
        for key, tensor in tensor_dict.items():
            print(f"- {key}: {tuple(tensor.shape)}, {tensor.dtype}, device: {tensor.device}")

    return tensor_dict


if __name__ == "__main__":
    # 1. Create an example tensor dictionary
    tensor_dict = {
        "softmax1": torch.rand(5),
        "softmax2": torch.rand(5),
        "softmax3": torch.rand(5),
    }

    # 3. Load for verification
    loaded_dict = load_distributions(save_path)
