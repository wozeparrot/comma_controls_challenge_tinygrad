from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import DEBUG, get_child, CI, OSX
import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype

def is_dtype_supported(dtype, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16: return False
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  if dtype == dtypes.half: return not (CI and device in {"GPU", "LLVM", "CUDA"})
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

# src: onnx/mapping.py
# not supported: STRING = 8 COMPLEX64 = 14, COMPLEX128 = 15
# NOTE: 17, 18, 19, 20 are float8, 10 is half
DTYPE_MAP = {1:dtypes.float, 2:dtypes.uint8, 3:dtypes.int8, 4:dtypes.uint16, 5:dtypes.int16, 6:dtypes.int32, 7:dtypes.int64,
              9:dtypes.bool, 10:dtypes.float, 11:dtypes.double, 12:dtypes.uint32, 13:dtypes.uint64, 16:dtypes.bfloat16,
              17:dtypes.float, 18:dtypes.float, 19:dtypes.float, 20:dtypes.float}

class SimModel:
  def __init__(self):
    self.vocab_size, self.dim, self.seq_len = 1024, 128, 20
    self.wt_embedding = nn.Linear(4, self.dim // 2)
    self.wt2_embedding = nn.Embedding(self.vocab_size, self.dim // 2)
    self.wp_embedding = nn.Embedding(self.seq_len, self.dim)
    self.h = [TransformerBlock(self.dim, 4, self.seq_len) for _ in range(4)]
    self.layer_norm_f = nn.LayerNorm(self.dim)
    self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

  def __call__(self, states:Tensor, tokens:Tensor) -> Tensor:
    if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, self.seq_len, device=states.device).reshape(1, -1).realize()
    tok_emb = self.wt_embedding(states)
    tok2_emb = self.wt2_embedding(tokens)
    tok_emb = tok_emb.cat(tok2_emb, dim=-1)
    pos_emb = self.wp_embedding(self.allpos)

    h = tok_emb + pos_emb
    h = h.sequential(self.h)

    logits = self.lm_head(self.layer_norm_f(h))
    return logits

  def load_from_onnx(self, onnx_model: onnx.ModelProto):
    def buffer_parse(inp: onnx.TensorProto) -> Tensor:
      if inp.data_type in (8,14,15): raise Exception(f"data type not supported {inp.name} {inp.dims} {inp.data_type}")
      dtype = DTYPE_MAP[inp.data_type] if is_dtype_supported(DTYPE_MAP[inp.data_type]) else dtypes.float32
      if dat := list(inp.float_data) or list(inp.int32_data) or list(inp.int64_data):
        return Tensor(dat, dtype=dtype).reshape(tuple(inp.dims))
      if len(inp.raw_data) > 0:
        return Tensor(np.frombuffer(inp.raw_data, dtype=tensor_dtype_to_np_dtype(inp.data_type)).astype(_to_np_dtype(dtype)).copy()).reshape(tuple(inp.dims))
      return Tensor(None)

    weights = {inp.name: buffer_parse(inp) for inp in onnx_model.graph.initializer}
    for k, v in weights.items():
      # remove transformer. from k
      k = k.lstrip("transformer.")

      # filtering
      if "attn.bias" in k: continue

      # transpose matmul weights
      if "x::MatMul" in k: v = v.transpose()

      # conversions
      if k == "x::MatMul_540": k = "wt_embedding.weight"
      for i, s in enumerate([542, 549, 556, 563]):
        if k == f"x::MatMul_{s+0}": k = f"h.{i}.attn.c_attn.weight"
        if k == f"x::MatMul_{s+4}": k = f"h.{i}.attn.c_proj.weight"
        if k == f"x::MatMul_{s+5}": k = f"h.{i}.mlp.c_fc.weight"
        if k == f"x::MatMul_{s+6}": k = f"h.{i}.mlp.c_proj.weight"
      if k == "x::MatMul_570": k = "lm_head.weight"

      # replace weight
      if DEBUG >= 2: print(f"{k}: {get_child(self, k).shape} {v.shape}")
      get_child(self, k).replace(v).realize()

class Attention:
  def __init__(self, dim, n_heads, seq_len):
    self.layer_norm = nn.LayerNorm(dim)
    self.c_attn = nn.Linear(dim, 3*dim, bias=False)
    self.c_proj = nn.Linear(dim, dim, bias=False)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.seq_len = seq_len

  def __call__(self, x:Tensor):
    x = self.layer_norm(x)
    xqkv = self.c_attn(x)
    xq, xk, xv = xqkv.chunk(3, dim=2)
    xq, xk, xv = xq.reshape(None, None, self.n_heads, self.head_dim), xk.reshape(None, None, self.n_heads, self.head_dim), xv.reshape(None, None, self.n_heads, self.head_dim)

    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(xk, xv, is_causal=True).transpose(1, 2).reshape(x.shape)
    return self.c_proj(attn)

class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.layer_norm = nn.LayerNorm(dim)
    self.c_fc = nn.Linear(dim, hidden_dim, bias=False)
    self.c_proj = nn.Linear(hidden_dim, dim, bias=False)

  def __call__(self, x:Tensor):
    return self.c_proj(self.c_fc(self.layer_norm(x)).gelu())

class TransformerBlock:
  def __init__(self, dim, n_heads, seq_len):
    self.attn = Attention(dim, n_heads, seq_len)
    self.mlp = FeedForward(dim, 4*dim)

  def __call__(self, x:Tensor):
    h = x + self.attn(x)
    return h + self.mlp(h)
