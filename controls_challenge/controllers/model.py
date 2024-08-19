import sys
sys.path.append("../")

from . import BaseController
from model import Model
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import load_state_dict, safe_load

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.model = Model()
    load_state_dict(self.model, safe_load("../ckpts/model_9.safetensors"))
    self.lataccel_history = []
    self.state_history = []
    self.action_history = []

  @TinyJit
  def _run_model(self, state, actions, current_lataccel, target_lataccel, future):
    return self.model(state, actions, current_lataccel, target_lataccel, future)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.lataccel_history.append(current_lataccel)
    self.state_history.append([state.roll_lataccel, state.v_ego, state.a_ego])

    if len(self.state_history) > 20:
      state_input = Tensor([self.state_history[-20:]])
      # actions_input = Tensor([self.action_history[-20:]]).unsqueeze(-1)
      actions_input = Tensor([[self.action_history[-1]]])
      current_lataccel_input = Tensor([self.lataccel_history[-20:]]).unsqueeze(-1)
      target_lataccel_input = Tensor([[target_lataccel]])
      # future_input = Tensor([future_plan]).transpose(1, 2)[:, :20]
      future_input = Tensor.zeros(1, 20, 4)
      # pad future_input
      future_input = future_input.pad((None, (0, 20 - future_input.shape[1]), None)).contiguous().realize()
      pred = self._run_model(state_input, actions_input, current_lataccel_input, target_lataccel_input, future_input).item()
      self.action_history.append(pred)
      return pred
    else:
      self.action_history.append(0.0)
      return 0.0
