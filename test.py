import onnx
from tinygrad.extra.onnx import get_run_onnx
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.helpers import getenv, GlobalCounters

from sim_model import SimModel

onnx_model = onnx.load("./controls_challenge/models/tinyphysics.onnx")
sim_model = SimModel()
sim_model.load_from_onnx(onnx_model)

output = sim_model(
  Tensor.randn(1, 20, 4, dtype=dtypes.float32),
  Tensor.full((1, 20), 500, dtype=dtypes.int32),
)
print(output.numpy())
