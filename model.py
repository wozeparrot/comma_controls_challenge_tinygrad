import warnings
from typing import Callable
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import round_up

class Model:
  def __init__(self):
    self.states_proj = nn.Linear(3, 32)
    self.states_ln = nn.LayerNorm(32)
    self.actions_proj = nn.Linear(1, 32)
    self.actions_ln = nn.LayerNorm(32)
    self.history_proj = nn.Linear(1, 32)
    self.history_ln = nn.LayerNorm(32)
    self.target_proj = nn.Linear(1, 32)
    self.target_ln = nn.LayerNorm(32)
    self.future_proj = nn.Linear(4, 128)
    self.future_ln = nn.LayerNorm(128)

    self.blocks = [Block(i, 4, 128) for i in range(4)]

    self.output = nn.Linear(128, 1)

  def __call__(self, states:Tensor, actions:Tensor, lataccel_history:Tensor, target_lataccel:Tensor, future:Tensor) -> Tensor:
    # projection
    states = self.states_ln(self.states_proj(states))
    actions = self.actions_ln(self.actions_proj(actions)).unsqueeze(1).expand(states.shape)
    lataccel_history = self.history_ln(self.history_proj(lataccel_history))
    target_lataccel = self.target_ln(self.target_proj(target_lataccel)).unsqueeze(1).expand(states.shape)
    future = self.future_ln(self.future_proj(future))

    # combine
    x = Tensor.cat(states, actions, lataccel_history, target_lataccel, dim=-1)
    x = x.cat(future, dim=1)

    # run thru blocks
    x = x.sequential(self.blocks)

    # output
    return self.output(x[:, -1, :]).tanh() * 2

class Block:
  def __init__(self, i, n_blocks, dim):
    self.ln1 = nn.LayerNorm(dim)
    self.time_mix = TimeMix(i, n_blocks, dim, 4)
    self.ln2 = nn.LayerNorm(dim)
    self.channel_mix = ChannelMix(i, n_blocks, dim)

  def __call__(self, x:Tensor) -> Tensor:
    x = x + self.time_mix.forward(self.ln1(x))
    x = x + self.channel_mix.forward(self.ln2(x))
    return x

class TimeMix:
  def __init__(self, i, n_blocks, dim, n_heads, *, linear:Callable=nn.Linear):
    self.n_heads, self.head_dim, self.head_divisor, self.mix_dim, self.decay_dim = n_heads, dim // n_heads, 8, 32, 64

    ratio_0_to_1 = (i / (n_blocks - 1)) if n_blocks != 1 else 1
    ratio_1_to_almost_0 = 1 - (i / n_blocks)
    self.time_maa_x = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_maa_w = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_maa_k = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_maa_v = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0).add(0.3 * ratio_0_to_1)
    self.time_maa_r = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(0.5 * ratio_1_to_almost_0)
    self.time_maa_u = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0).add(0.3 * ratio_0_to_1)
    self.time_maa_g = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(0.5 * ratio_1_to_almost_0)

    self.time_maa_w1 = Tensor.zeros(dim, self.mix_dim * 6)
    self.time_maa_w2 = Tensor.uniform(6, self.mix_dim, dim, low=-0.01, high=0.01)

    self.time_decay_w1 = Tensor.zeros(dim, self.decay_dim)
    self.time_decay_w2 = Tensor.uniform(self.decay_dim, dim, low=-0.01, high=0.01)
    self.time_decay = (-6 + 5 * Tensor.arange(dim).div(dim - 1).pow(0.7 + 1.3 * ratio_0_to_1)).reshape(n_heads, self.head_dim)

    self.time_ualue_w1 = Tensor.zeros(dim, self.decay_dim)
    self.time_ualue_w2 = Tensor.uniform(self.decay_dim, dim, low=-0.01, high=0.01)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, dim, bias=False)
    self.value = linear(dim, dim, bias=False)
    self.output = linear(dim, dim, bias=False)
    self.gate = linear(dim, dim, bias=False)
    self.ln_x = nn.LayerNorm(dim)

  @staticmethod
  def wkv(r:Tensor, k:Tensor, v:Tensor, w:Tensor, kv_state:Tensor, C:int=28 if dtypes.default_float == dtypes.float32 else 14):
    B, H, T, X = r.shape
    if T % C != 0:
      warnings.warn(f"T % C != 0, T={T}, C={C}, T % C={T % C}", RuntimeWarning)
      # try to find the nearest C
      if T % 2 != 0: C = 1
      else:
        while T % C != 0: C -= 2
      warnings.warn(f"new C={C}, N={T // C}", RuntimeWarning)
    N = T // C

    if T == 1:
      y = kv_state
      kv_state = kv_state * w.transpose(-2, -1) + (k.transpose(-2, -1) @ v)
      return r @ y, kv_state
    else:
      w_log = w.maximum(0.005).float().log()
      wc_log = w_log.reshape(w.shape[0], H, N, C, X)
      wc_log_cumsum = wc_log.cumsum(axis=-2)

      shifted_wc_log_cumsum = wc_log_cumsum.pad2d((0, 0, 1, -1))

      ws = wc_log.sum(axis=-2, keepdim=True)
      w_inter = ws - wc_log_cumsum
      w_intra = wc_log_cumsum - wc_log

      ws = list(map(lambda x: x.squeeze(-3), ws.transpose(-2, -1).exp().split(1, dim=-3)))
      w_inter = w_inter.exp()
      w_intra = w_intra.exp()

      r, k, v = r.reshape(B, H, N, C, X), k.reshape(B, H, N, C, X), v.reshape(B, H, N, C, X)

      wc_log_offset = shifted_wc_log_cumsum[..., C//2:C//2 + 1, :]
      r_decay = (shifted_wc_log_cumsum - wc_log_offset).exp()
      k_inv_decay = (wc_log_offset - wc_log_cumsum).exp()
      a = ((r * r_decay) @ (k * k_inv_decay).transpose(-2, -1)).tril(-1)
      out = a @ v

      wkv = (k * w_inter).transpose(-2, -1) @ v
      wkv = list(map(lambda x: x.squeeze(-3), wkv.split(1, dim=-3)))

      states = []
      for i in range(T // C):
        states.append(kv_state)
        kv_state = kv_state * ws[i] + wkv[i]
      states = Tensor.stack(*states, dim=2)

      out = out + (r * w_intra) @ states
      out = out.reshape(B, H, T, X)
      return out.cast(dtypes.default_float), kv_state.cast(dtypes.default_float)

  def __call__(self, x:Tensor, state:Tensor | None):
    (B, T, D), H, X = x.shape, self.n_heads, self.head_dim

    # token shift
    xx = x.pad((None, (1, 0), None)).shrink((None, (0, x.shape[1]), None)) if state is None else state[0]

    # lora time mix
    xxx = x.lerp(xx, self.time_maa_x)
    xxx = xxx.matmul(self.time_maa_w1).tanh().reshape(B*T, 6, -1).T
    xxx = xxx.matmul(self.time_maa_w2).reshape(6, B, T, -1)
    mw, mk, mv, mr, mu, mg = list(map(lambda x: x.squeeze(0), xxx.split(1, dim=0)))
    xr, xw, xk, xv, xu, xg = x.lerp(xx, self.time_maa_r + mr), x.lerp(xx, self.time_maa_w + mw), x.lerp(xx, self.time_maa_k + mk), x.lerp(xx, self.time_maa_v + mv), x.lerp(xx, self.time_maa_u + mu), x.lerp(xx, self.time_maa_g + mg)

    # projection
    r, k, v = self.receptance(xr), self.key(xk), self.value(xv)
    r, k, v = r.reshape(B, T, H, X), k.reshape(B, T, H, X), v.reshape(B, T, H, X)
    r, k, v = r.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # lora time decay
    w = self.time_decay.reshape(1, H, 1, X).float()
    w = w + xw.matmul(self.time_decay_w1).tanh().matmul(self.time_decay_w2).reshape(B, T, H, X).transpose(1, 2)
    w = w.exp().neg().exp()

    # scale k
    k = k * (1 - w)

    # lora boost
    u = self.value(xu).reshape(B, T, H, X).transpose(1, 2)
    u = u + xu.matmul(self.time_ualue_w1).tanh().matmul(self.time_ualue_w2).reshape(B, T, H, X).transpose(1, 2)

    # wkv
    out, kv_state = [], Tensor.zeros(B, H, X, X, dtype=x.dtype, device=x.device) if state is None else state[1].reshape(B, H, X, X)
    out, kv_state = TimeMix.wkv(r, k, v, w, kv_state)

    # add boost
    out = out + u

    # project
    out = self.ln_x(out.transpose(1, 2).reshape(B, T, D))
    out = self.output(out * self.gate(xg).silu())
    return out if state is None else (out, x, kv_state.reshape(B, 1, H * X * X))
  def forward(self, x):
    T = x.shape[1]
    if x.shape[1] % 2 != 0: x = x.pad((None, (0, 1), None))
    return self(x, None)[:, :T, :]

class ChannelMix:
  def __init__(self, i, n_blocks, dim, *, linear:Callable=nn.Linear):
    ratio_1_to_almost_0 = 1 - (i / n_blocks)
    self.time_maa_k = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)
    self.time_maa_r = 1 - Tensor.arange(dim).div(dim).reshape(1, 1, dim).pow(ratio_1_to_almost_0)

    self.receptance = linear(dim, dim, bias=False)
    self.key = linear(dim, round_up(int(dim * 3.5), 32), bias=False)
    self.value = linear(round_up(int(dim * 3.5), 32), dim, bias=False)

  def __call__(self, x:Tensor, state:Tensor | None):
    # token shift
    xx = x.pad((None, (1, 0), None)).shrink((None, (0, x.shape[1]), None)) if state is None else state[0]
    xr, xk = x.lerp(xx, self.time_maa_r), x.lerp(xx, self.time_maa_k)

    # projection and activation
    k = self.key(xk).relu().square()
    kv = self.value(k)

    # gate
    out = self.receptance(xr).sigmoid() * kv

    return out if state is None else (out, x)
  def forward(self, x): return self(x, None)
