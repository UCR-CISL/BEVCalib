import os

import torch


def load_checkpoint(model, ckpt_path, device="cpu"):
    """Load BEVCalib weights into `model`.

    Accepts:
      - a Hugging Face checkpoint directory (containing `model.safetensors`), e.g. the
        `--local-dir` of `huggingface-cli download cisl-hf/BEVCalib --revision ...`
      - a `.safetensors` file
      - a `.pth` checkpoint saved by the training scripts (`{"model_state_dict": ...}`)
    """
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.safetensors")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not ckpt_path.endswith(".safetensors"):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model

    from safetensors.torch import load_file

    state_dict = load_file(ckpt_path, device=str(device))
    # Some modules are registered under two names (e.g. SparseBasicBlock.conv1 and
    # SparseBasicBlock.bottle_neck.0), so their tensors are shared. safetensors stores each
    # shared tensor only once, so the duplicate names can be absent from the file. A missing
    # key is fine only if another name for the same tensor was loaded.
    names_by_tensor = {}
    for name, tensor in model.state_dict().items():
        names_by_tensor.setdefault(tensor.data_ptr(), []).append(name)
    loaded = set(state_dict)

    result = model.load_state_dict(state_dict, strict=False)
    model_state = model.state_dict()
    truly_missing = [
        k for k in result.missing_keys
        if not any(alias in loaded for alias in names_by_tensor[model_state[k].data_ptr()])
    ]
    if truly_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading {ckpt_path}:\n"
            f"    Missing key(s): {truly_missing}\n"
            f"    Unexpected key(s): {result.unexpected_keys}"
        )
    return model

