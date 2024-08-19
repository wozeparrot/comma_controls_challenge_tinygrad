import time, math

from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv, GlobalCounters, round_up, dedup
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tqdm import tqdm
import wandb
import onnx

from model import Model
from sim_model import SimModel
from dataloader import batch_load, get_train_files, load_single_file, BatchDesc, CONTEXT_LENGTH, ROLLOUT_LENGTH, FUTURE_PLAN_STEPS

if __name__ == "__main__":
  Tensor.no_grad = False
  Tensor.training = True

  wandb.init(project="tinygrad-comma-controls-challenge")

  GPUS = tuple([f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))])
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]

  # config hyperparameters
  BS = getenv("BS", 64) * len(GPUS)
  EVAL_BS = getenv("EVAL_BS", 8) * len(GPUS)
  EPOCHS = getenv("EPOCHS", 10)
  WARMUP_STEPS = getenv("WARMUP_STEPS", 400)
  WARMUP_LR = getenv("WARMPUP_LR", 1e-5)
  START_LR = getenv("START_LR", 1e-5)
  END_LR = getenv("END_LR", 1e-5)
  B1 = getenv("B1", 0.9)
  B2 = getenv("B2", 0.98)
  WD = getenv("WD", 0.01)
  MOMENTUM = getenv("MOMENTUM", 0.98)
  FACTOR = getenv("FACTOR", 2)

  # log config
  wandb.config.update({
    "GPUS": len(GPUS),
    "BS": BS,
    "EPOCHS": EPOCHS,
    "WARMUP_STEPS": WARMUP_STEPS,
    "WARMUP_LR": WARMUP_LR,
    "START_LR": START_LR,
    "END_LR": END_LR,
    "B1": B1,
    "B2": B2,
    "WD": WD,
    "MOMENTUM": MOMENTUM,
    "FACTOR": FACTOR,
  })

  # initialize models
  model = Model()
  sim_model = SimModel()
  sim_model.load_from_onnx(onnx.load("./controls_challenge/models/tinyphysics.onnx"))
  # shard model
  for k, v in get_state_dict(model).items():
    v.realize().to_(GPUS)
  for k, v in get_state_dict(sim_model).items():
    v.realize().to_(GPUS)
  # load_state_dict(model, safe_load("ckpts/model_0.safetensors"))

  optim = AdamW(get_parameters(model), lr=WARMUP_LR, b1=B1, b2=B2, weight_decay=WD)

  def data_get(it):
    x, cookie = next(it)
    return x["X"].shard(GPUS, axis=0), x["future"].shard(GPUS, axis=0), cookie
  steps_in_train_epoch = round_up(len(get_train_files()), BS) // BS

  warming_up = True
  def get_lr(step: int) -> float:
    global warming_up
    if warming_up:
      lr = START_LR * (step / WARMUP_STEPS) + WARMUP_LR * (1 - step / WARMUP_STEPS)
      if step >= WARMUP_STEPS: warming_up = False
    else: lr = END_LR + 0.5 * (START_LR - END_LR) * (1 + math.cos(((step - WARMUP_STEPS) / ((EPOCHS * steps_in_train_epoch) - WARMUP_STEPS)) * math.pi))
    return lr

  @TinyJit
  def train_step(x: Tensor, future: Tensor, lr: Tensor):
    # unpack x
    states, target_lataccel, actions = x.split([3, 1, 1], dim=-1)

    # get action thru model
    loss = 0
    lataccel_history = target_lataccel[:, :20, :].contiguous()
    actions_history = Tensor.zeros(BS, 20, 1, dtype=dtypes.float32).shard(GPUS, axis=0)
    for i in range(20):
      states_ = states[:, i:i+20, :]
      action = model(states_, actions_history[:, -1, :].detach(), lataccel_history.detach(), target_lataccel[:, i+19, :], future[:, i:i+20, :])
      actions_history = actions_history[:, 1:, :].detach().cat(action.unsqueeze(1), dim=1)

      # run action thru simulator model
      lataccel_tokens = (((lataccel_history / 5) * (1024 // 2)) + 512).clamp(0, 1023).ceil().cast(dtypes.int32).squeeze(-1)
      logits = sim_model(actions_history.cat(states_, dim=-1), lataccel_tokens.detach())

      last_logits = logits[:, -1, :].div(0.1).softmax()
      lataccel_token = last_logits.argmax(-1, keepdim=True)
      # straight thru estimator
      lataccel_token = lataccel_token.detach() + (last_logits - last_logits.detach())
      lataccel_token = lataccel_token[:, 0].unsqueeze(-1)

      lataccel = ((lataccel_token.cast(dtypes.float32) - 512) * 5 / (1024 // 2)).unsqueeze(-1)
      lataccel_history = lataccel_history[:, 1:, :].cat(lataccel, dim=1)

      # loss toward target lataccel
      target_lataccel_tokens = (((target_lataccel[:, i+19, 0] / 5) * (1024 // 2)) + 512).clamp(0, 1023).ceil().cast(dtypes.int32)
      loss = loss + logits[:, -1].sparse_categorical_crossentropy(target_lataccel_tokens)
    loss = loss / 20

    # jerk cost
    jerk_loss = ((lataccel[:, 0, 0] - lataccel_history[:, -2, 0]) / 0.1).square().mean()
    loss = loss + jerk_loss

    # lataccel_loss = (target_lataccel[:, i+19, 0] - lataccel[:, 0, 0]).square().mean()

    optim.lr.assign(lr)
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss

  step = 0
  for e in range(EPOCHS):
    # train epoch
    batch_loader = batch_load(descs={
      "X": BatchDesc(shape=(CONTEXT_LENGTH+ROLLOUT_LENGTH, 5), dtype=dtypes.float32),
      "future": BatchDesc(shape=(FUTURE_PLAN_STEPS+ROLLOUT_LENGTH, 4), dtype=dtypes.float32),
    }, load_single_fn=load_single_file, files_fn=get_train_files, bs=BS)
    it = iter(tqdm(batch_loader, total=steps_in_train_epoch, desc=f"epoch {e}"))
    i, proc = 0, data_get(it)
    st, prev_cookies = time.perf_counter(), []
    while proc is not None:
      GlobalCounters.reset()

      lr = get_lr(step)
      loss, proc = train_step(proc[0], proc[1], Tensor([lr], dtype=dtypes.float32).to(GPUS)), proc[2]
      pt = time.perf_counter()

      if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []
      try: next_proc = data_get(it)
      except StopIteration: next_proc = None
      dt = time.perf_counter()

      loss = loss.item()
      at = time.perf_counter()

      wandb.log({
        "epoch": e + (i + 1) / steps_in_train_epoch,
        "train/lr": lr,
        "train/loss": loss,

        "train/step_time": at - st,
        "train/python_time": pt - st,
        "train/data_time": dt - pt,
        "train/accel_time": at - dt,

        "train/gb": GlobalCounters.mem_used / 1e9,
        "train/gbps": GlobalCounters.mem_used * 1e-9 / (at - st),
        "train/gflops": GlobalCounters.global_ops * 1e-9 / (at - st),
      })
      tqdm.write(
        f"{e:7} - lr: {lr:8.8f}, loss: {loss:6.6f}, "
        f"{((at - st)) * 1000.0:7.2f} ms step, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms data, {(at - dt) * 1000.0:7.2f} ms accel, "
        f"{GlobalCounters.mem_used / 1e9:7.2f} GB used, {GlobalCounters.mem_used * 1e-9 / (at - st):9.2f} GB/s, {GlobalCounters.global_ops * 1e-9 / (at - st):9.2f} GFLOPS"
      )

      st = at
      prev_cookies.append(proc)
      proc, next_proc = next_proc, None
      i += 1
      step += 1

    # save epoch checkpoint
    safe_save(get_state_dict(model), f"ckpts/model_{e}.safetensors")
