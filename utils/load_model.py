import torch
from model import ModelConfig, MyGPT

def build_optimizer(model: MyGPT, weight_decay: float, lr_max: float) -> torch.optim.AdamW:
    decay_params     = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    n_decay   = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay:,} params with decay, {n_nodecay:,} without.")
    return torch.optim.AdamW(param_groups, lr=lr_max, betas=(0.9, 0.95), eps=1e-8)

def load_model(
    ckpt_path: str, 
    device: str, 
    weight_decay: float | None = None, 
    lr_max: float | None = None
) -> MyGPT:
    # PyTorch 2.6+ defaults weights_only=True, which blocks custom classes.
    # Allowlisting ModelConfig is the safe fix — avoids arbitrary code execution
    # while still supporting our known dataclass.
    torch.serialization.add_safe_globals([ModelConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    config = ckpt['config']
    model  = MyGPT(config).to(device)
    # Strip _orig_mod. prefix if checkpoint was saved from a torch.compile()d model
    state_dict = ckpt["model"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    global_step = ckpt.get('global_step', '?')

    print(f"Loaded checkpoint: {ckpt_path}")

    if weight_decay is not None:
        optimizer = build_optimizer(model, weight_decay, lr_max)
        optimizer.load_state_dict(ckpt['optimizer'])
        return model, optimizer, global_step
    else:
        return model