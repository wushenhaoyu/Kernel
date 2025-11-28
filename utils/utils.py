import hashlib
import inspect
import os
import sys
import torch
import time
import re
import torch
import torch.nn as nn
import importlib.util as _imu

from typing import Any, Tuple, List
from pathlib import Path
from multiprocessing import get_context


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
    
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> bool:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:  
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return True

    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False
    

def load_models_and_inputs(
    ref_py: Path,
    generated_py: Path,
) -> Tuple[nn.Module, nn.Module, List[torch.Tensor]]:


    ref_spec = _imu.spec_from_file_location("ref_mod", ref_py)
    ref_mod = _imu.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref_mod)

    RefModel = getattr(ref_mod, "Model", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    if RefModel is None or get_inputs is None:
        raise RuntimeError("ref.py must define Model 与 get_inputs()")

    gen_spec = _imu.spec_from_file_location("gen_mod", generated_py)
    gen_mod = _imu.module_from_spec(gen_spec)
    gen_spec.loader.exec_module(gen_mod)
    ModelNew = getattr(gen_mod, "ModelNew", None)
    if ModelNew is None:
        raise RuntimeError("generated.py must define ModelNew")

    inputs = get_inputs()  
    #if isinstance(inputs, dict):
    #    inputs = [inputs]
    #if not isinstance(inputs, (list, tuple)):
    #    inputs = [inputs]

    ref_model = RefModel()
    test_model = ModelNew()

    return ref_model, test_model, inputs

def compile_kernel(py_path: Path) -> Tuple[bool, str]:
    name = f"kernel_{hashlib.md5(py_path.read_bytes()).hexdigest()}"
    try:
        spec = _imu.spec_from_file_location(name, py_path)
        mod  = _imu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True, ""
    except Exception as e:
        return False, str(e)




def correctness_and_benchmark(
    ref_model: torch.nn.Module,
    test_model: torch.nn.Module,
    inputs: List[torch.Tensor],
    device: torch.device,
    tol: float = 1e-3,
    warmup: int = 5,
    repeat: int = 20
) -> Tuple[bool, float, float, float, float]:

    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    def _first_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
        raise RuntimeError("Moudle cannot find Tensor")

    with torch.no_grad():
        ref_out = ref_model(*inputs)
        test_out = test_model(*inputs)

        ref_t = _first_tensor(ref_out)
        test_t = _first_tensor(test_out)

        diff = (test_t - ref_t).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        passed = torch.allclose(ref_t, test_t, atol=tol, rtol=tol)

    for _ in range(warmup):
        ref_model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ref_times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ref_times.append((time.perf_counter() - t0) * 1000)

    for _ in range(warmup):
        test_model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    test_times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        test_model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        test_times.append((time.perf_counter() - t0) * 1000)

    ref_avg = sum(ref_times) / len(ref_times)
    test_avg = sum(test_times) / len(test_times)

    return passed, max_err, mean_err, ref_avg, test_avg

def strip_fence(code: str) -> str:
    code = code.strip()
    pattern = re.compile(
        r'^```(?:python|py)?\n(.*?)```$',
        re.MULTILINE | re.DOTALL
    )
    match = pattern.fullmatch(code)
    if match:
        return match.group(1).strip()
    return code



if __name__ == "__main__":
    PY = Path(r"D:\data\design\code\Kernel\runs\deepseek_deepseek-chat_oneshot_20251128_140315\level1\1_Square_matrix_multiplication_\round_000\code\generated.py")
    ok, log = compile_kernel(PY)
    print("编译结果:", "✅ 成功" if ok else "❌ 失败")
    if not ok:
        print("日志:", log)
    