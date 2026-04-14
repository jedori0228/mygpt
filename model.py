from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:

    # Token
    DToken: int = 32768
    """Number of vocabulary"""

    # Context
    ContextSize: int = 256
    """The maximum number of tokens in a single context"""

    # Head dimension
    DHead: int = 64
    """Dimension of the Head"""
    NHeadQ: int = 6
    """Number of heads for query"""
    NHeadKV: int = 6
    """Number of heads for Key/Value. Can be smaller than NHeadQ (GQA)"""
    @property
    def DEmbed(self) -> int:
        """Dimension of embedding; always HeadSize times NHeadQ"""
        return self.DHead * self.NHeadQ

    NLayer: int = 6
    """Number of Layer"""

# TODO
def ApplyRoPE(x, cos_sin_RoPE):

    # x: (Batch, Context, NHead, Embedding)
    # print('[ApplyRoPE] x.shape', x.shape)

    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos_sin_RoPE[0] + x2 * cos_sin_RoPE[1] # rotate pairs of dims
    y2 = x1 * (-cos_sin_RoPE[1]) + x2 * cos_sin_RoPE[0]
    return torch.cat([y1, y2], 3)


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int):

        """
        SelfAttention init

        Args:
            config (ModelConfig): ModelConfig object
            layer_idx (int): Index of this layer
        """

        super().__init__()

        if config.NHeadQ % config.NHeadKV != 0:
            raise ValueError(f"NHeadQ ({config.NHeadQ}) must be divisible by NHeadKV ({config.NHeadKV}) for GQA.")

        self.cfg = config
        self.LayerIndex = layer_idx

        self.W_q = nn.Linear(config.DEmbed, config.NHeadQ * config.DHead, bias=False)
        self.W_k = nn.Linear(config.DEmbed, config.NHeadKV * config.DHead, bias=False)
        self.W_v = nn.Linear(config.DEmbed, config.NHeadKV * config.DHead, bias=False)

        self.W_o = nn.Linear(config.DEmbed, config.DEmbed, bias=False)

    def forward(self, x, cos_sin_RoPE):

        """
        SelfAttention forward

        Args:
            x: Input
            R_RoPE: Pre-calculated 
        """

        InShape_Batch, InShape_Context, InShape_Emb = x.shape

        # q: (InShape_Batch, InShape_Context, NHeadQ*DHead)
        # -> InShape_Batch, InShape_Context, NHeadQ, DHead)

        q = self.W_q(x).view(InShape_Batch, InShape_Context, self.cfg.NHeadQ, self.cfg.DHead)
        k = self.W_k(x).view(InShape_Batch, InShape_Context, self.cfg.NHeadKV, self.cfg.DHead)
        v = self.W_v(x).view(InShape_Batch, InShape_Context, self.cfg.NHeadKV, self.cfg.DHead)

        # Apply RoPE
        q = ApplyRoPE(q, cos_sin_RoPE)
        k = ApplyRoPE(k, cos_sin_RoPE)

        # o: (InShape_Batch, InShape_Context, )
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o = o.contiguous().view(InShape_Batch, InShape_Context, -1)

        # Projection
        o = self.W_o(o)

        return o

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):

        """
        Multi-layer Perceptron init

        Args:
            config (ModelConfig): ModelConfig object
        """

        super().__init__()

        DIntermediate = 4 * config.DEmbed 
        
        self.W1 = nn.Linear(config.DEmbed, DIntermediate, bias=False)
        self.act = nn.GELU()
        self.W2 = nn.Linear(DIntermediate, config.DEmbed, bias=False)

    def forward(self, x):
        x = self.W1(x)
        x = self.act(x)
        x = self.W2(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int):

        """
        Block init

        Args:
            config (ModelConfig): ModelConfig object
            layer_idx (int): Index of this layer
        """

        super().__init__()
        self.attn = SelfAttention(config, layer_idx)
        self.ln_attn = nn.LayerNorm(config.DEmbed)

        self.mlp = MLP(config)
        self.ln_mlp = nn.LayerNorm(config.DEmbed)

    def forward(self, x, cos_sin_RoPE):
        x = x + self.attn(self.ln_attn(x), cos_sin_RoPE)
        x = x + self.mlp(self.ln_mlp(x))
        return x

class MyGPT(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.cfg = config
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.DToken, config.DEmbed),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.NLayer)]),
        })
        self.ln_te = nn.LayerNorm(config.DEmbed)
        self.ln_bl = nn.LayerNorm(config.DEmbed)

        self.lm_head = nn.Linear(config.DEmbed, config.DToken, bias=False)

        # RoPE cos and sin are precalculated
        arr_RoPE_cos, arr_RoPE_sin = self.PrecomputeRoPE(config.ContextSize * 10, config.DHead)
        self.register_buffer("arr_RoPE_cos", arr_RoPE_cos, persistent=False)
        self.register_buffer("arr_RoPE_sin", arr_RoPE_sin, persistent=False)

    def get_device(self):
        return self.transformer.wte.weight.device

    def PrecomputeRoPE(self, seq_len, head_dim, base=10000):

        """
        Source: https://arxiv.org/pdf/2104.09864
        """

        # Use the model's device
        device = self.transformer.wte.weight.device
        
        # theta = 1/Base**(2j/DHead), j = [0, 1, 2, ..., d/2]
        # First make "2j" = [0, 2, 4, ..., ]
        arr_2j = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # Then make "theta"
        # arr_theta: d/2-length array
        arr_theta = 1.0 / (base ** (arr_2j / head_dim))

        # Then make the position array
        # arr_Pos: seq_len-length array
        arr_Pos = torch.arange(seq_len, dtype=torch.float32, device=device)

        # arr_PosTheta: (seq_len, d/2) matrix
        # arr_PosTheta[m, i] = m * theta_i; use torch.outer
        arr_PosTheta = torch.outer(arr_Pos, arr_theta)

        arr_cos = torch.cos(arr_PosTheta)
        arr_sin = torch.sin(arr_PosTheta)

        # add batch and head dims for later broadcasting
        arr_cos = arr_cos[None, :, None, :]
        arr_sin = arr_sin[None, :, None, :]

        return arr_cos, arr_sin

    def forward(self, x_token, targets=None):

        InShape_Batch, InShape_Context = x_token.shape

        cos_sin_RoPE = (self.arr_RoPE_cos[:, :InShape_Context], self.arr_RoPE_sin[:, :InShape_Context])

        # Token to Embedding
        x = self.transformer.wte(x_token)
        x = self.ln_te(x)

        for i_layer, block in enumerate(self.transformer.h):
            x = block(x, cos_sin_RoPE)

        x = self.ln_bl(x)

        # logits: (Batch, Context, DToken)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        else:
            # inference: just return the logits directly
            return logits

    
    @torch.inference_mode()
    def generate(self, input_tokens, max_tokens, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(input_tokens, list)
        device = self.get_device()

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        current_sequence = torch.tensor([input_tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            # forward
            logits = self.forward(current_sequence) # (B, T, vocab_size)
            # Last token
            logits = logits[:, -1, :] # (B, vocab_size)
            # softmax
            probs = F.softmax(logits, dim=-1)
            # throw next id(s)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            current_sequence = torch.cat((current_sequence, next_ids), dim=1)
        return current_sequence


# Test code
DoTest = False
if DoTest:

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    DToken = enc.n_vocab

    encode = lambda list_c: enc.encode(list_c)
    decode = lambda list_i: enc.decode(list_i)


    _config = ModelConfig()
    _config.DToken = DToken

    my_model = MyGPT(_config).to(device)

    test_input = torch.randint(low=0, high=DToken, size=(4, 16), device=device)
    
    # o = my_model(test_input)

    input_text = 'Dog'

    tmp_start_text = encode(input_text)

    # generate from the model
    print('Input text:')
    print(input_text)

    print()

    output_text = decode(my_model.generate(tmp_start_text, max_tokens=10)[0].tolist())
    print(output_text)

