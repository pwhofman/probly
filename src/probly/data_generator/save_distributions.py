# Save generated data as .pt format
from __future__ import annotations


def save_distributions(
    tensor_dict: Dict[str, Any],
    save_path: str,
    create_dir: bool = False,
    verbose: bool = True,
) -> None:
    """Save a tensor dictionary as a .pt file.

    Parameters:
        tensor_dict: A dictionary that contains tensors.
        save_path:   Target path (should end with .pt or .pth).
        create_dir:  Whether to automatically create the directory.
        verbose:     Whether to print detailed information.
    """
    # Check file suffix
    if not (save_path.endswith(".pt") or save_path.endswith(".pth")):
        save_path += ".pt"

    # Create directory (if needed)
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save dictionary
    torch.save(tensor_dict, save_path)

    if verbose:
        print(f"Tensor dict has been saved to: {save_path}")
        print("Dictionary overview:")
        total_size = 0
        for key, tensor in tensor_dict.items():
            size_mb = tensor.element_size() * tensor.nelement() / (1024**2)
            total_size += size_mb
            print(f"- {key}: {tuple(tensor.shape)}, {tensor.dtype}, {size_mb:.2f} MB")
        print(f"Total size: {total_size:.2f} MB")


if __name__ == "__main__":
    # 1. Create an example tensor dictionary
    tensor_dict = {
        "softmax1": torch.rand(5),
        "softmax2": torch.rand(5),
        "softmax3": torch.rand(5),
    }

    # 2. Save using the helper function
    save_path = "basic_example.pt"
    save_distributions(tensor_dict, save_path)
