# Based off the tinygrad mlperf dataloader: https://github.com/tinygrad/tinygrad/blob/master/examples/mlperf/dataloader.py

import glob, random, signal, os, contextlib
from multiprocessing import Queue, Process, shared_memory, cpu_count
from pathlib import Path
from typing import Callable
from dataclasses import dataclass

from tinygrad.helpers import Context, getenv, prod
from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType
import numpy as np
import pandas as pd
from tqdm import tqdm

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

ROLLOUT_LENGTH = FPS * 2  # 2 secs

BASE_PATH = Path(__file__).parent / "controls_challenge" / "data"
def get_train_files(): return glob.glob(str(BASE_PATH / "*.csv"))

# **** single load function ****

def load_single_file(file, return_bytes=True):
  df = pd.read_csv(file)
  processed_df = pd.DataFrame({
    "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
    "v_ego": df["vEgo"].values,
    "a_ego": df["aEgo"].values,
    "target_lataccel": df["targetLateralAcceleration"].values,
    "steer_command": -df["steerCommand"].values
  })

  # pick a step idx
  step_idx = random.randint(CONTEXT_LENGTH, 100)

  # current state
  curr_state = processed_df.iloc[step_idx-CONTEXT_LENGTH:step_idx+ROLLOUT_LENGTH]
  state = curr_state[["roll_lataccel", "v_ego", "a_ego"]].values
  target_lataccel = curr_state["target_lataccel"].values
  actions = curr_state["steer_command"].values

  # future plan
  future = processed_df.iloc[step_idx:step_idx+FUTURE_PLAN_STEPS+ROLLOUT_LENGTH]
  future = future[["target_lataccel", "roll_lataccel", "v_ego", "a_ego"]].values
  future = future.astype(np.float32)

  # stack together
  data = np.concatenate([state, target_lataccel.reshape(CONTEXT_LENGTH+ROLLOUT_LENGTH, 1), actions.reshape(CONTEXT_LENGTH+ROLLOUT_LENGTH, 1)], axis=-1)
  data = data.reshape((CONTEXT_LENGTH+ROLLOUT_LENGTH, 5)).astype(np.float32)

  return {
    "X": data.tobytes(),
    "future": future.tobytes()
  }

# **** semi generic dataloder ****

@dataclass
class BatchDesc:
  shape: tuple
  dtype: DType

def shuffled_indices(n:int):
  indices = {}
  for i in range(n-1, -1, -1):
    j = random.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in:Queue, q_out:Queue, load_single_fn:Callable, tensors:dict[str, Tensor]):
  signal.signal(signal.SIGINT, lambda *_: exit(0))
  with Context(DEBUG=0):
    while (recv := q_in.get()):
      idx, file = recv
      data = load_single_fn(file)

      # write to shared memory
      for name, t in tensors.items():
        t[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = data[name]

      q_out.put(idx)
    q_out.put(None)

def batch_load(descs: dict[str, BatchDesc], load_single_fn, files_fn, bs:int=32, shuffle=True):
  files = files_fn()
  BATCH_COUNT = min(32, len(files) // bs)

  gen = shuffled_indices(len(files)) if shuffle else iter(range(len(files)))
  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      file = files[next(gen)]
      q_in.put((idx, file))

  running = True
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if running:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while True:
      num = q_out.get()//bs
      gotten[num] += 1
      if gotten[num] == bs: break
    gotten[num] = 0
    return {name: tensors[name][num*bs:(num+1)*bs] for name in descs}, Cookie(num)

  q_in, q_out = Queue(), Queue()

  # get sizes
  szs = {name: (BATCH_COUNT*bs, *desc.shape) for name, desc in descs.items()}

  shms = {}
  for name, desc in descs.items():
    if os.path.exists(f"/dev/shm/dataloader_{name}"): os.unlink(f"/dev/shm/dataloader_{name}")
    shms[name] = shared_memory.SharedMemory(name=f"dataloader_{name}", create=True, size=prod(szs[name]) * desc.dtype.itemsize)

  procs = []
  try:
    tensors = {name: Tensor.empty(*szs[name], dtype=desc.dtype, device=f"disk:/dev/shm/dataloader_{name}") for name, desc in descs.items()}

    for _ in range(getenv("DATAPROC", 1)):
      p = Process(target=loader_process, args=(q_in, q_out, load_single_fn, tensors))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    for _ in range(0, len(files)//bs): yield receive_batch()
  finally:
    running = False
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    for p in procs: p.terminate()
    for p in procs: p.join()
    for shm in shms.values(): shm.close()
    for shm in shms.values():
      try: shm.unlink()
      except FileNotFoundError: pass

if __name__ == "__main__":
  BS = 1
  preprocessed_train_files = get_train_files()

  batch_loader = batch_load(descs={
    "X": BatchDesc(shape=(CONTEXT_LENGTH+ROLLOUT_LENGTH, 5), dtype=dtypes.float32),
    "future": BatchDesc(shape=(FUTURE_PLAN_STEPS+ROLLOUT_LENGTH, 4), dtype=dtypes.float32),
  }, load_single_fn=load_single_file, files_fn=get_train_files, bs=BS)
  x = next(batch_loader)[0]["X"]
  print(x.numpy())

  with tqdm(total=(len(preprocessed_train_files)//BS)*BS) as pbar:
    for x, _ in batch_load(descs={
      "X": BatchDesc(shape=(CONTEXT_LENGTH+ROLLOUT_LENGTH, 5), dtype=dtypes.float32),
      "future": BatchDesc(shape=(FUTURE_PLAN_STEPS+ROLLOUT_LENGTH, 4), dtype=dtypes.float32),
    }, load_single_fn=load_single_file, files_fn=get_train_files, bs=BS):
      pbar.update(x["X"].shape[0])
