import hashlib
import os
import torch
import time
import importlib.util

from typing import Tuple, List
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
    



def compile_kernel(py_path: Path) -> Tuple[bool, str]:
    name = f"kernel_{hashlib.md5(py_path.read_bytes()).hexdigest()}"
    try:
        spec = importlib.util.spec_from_file_location(name, py_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True, ""
    except Exception as e:
        return False, str(e)




def correctness_test(ref_model: torch.nn.Module,
                     test_model: torch.nn.Module,
                     device: torch.device,
                     tol: float = 1e-3) -> Tuple[bool, float, float]:

    import importlib.util
    import inspect

    file_path = inspect.getfile(ref_model.__class__)
    spec = importlib.util.spec_from_file_location("ref_mod", file_path)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)

    get_inputs = getattr(ref_mod, "get_inputs", None)
    if get_inputs is None:
        raise RuntimeError("ref_model not define function get_inputs()")

    inputs = get_inputs()                      
    if isinstance(inputs, dict):               
        inputs = [inputs]
    if not isinstance(inputs, (list, tuple)):  
        inputs = [inputs]

    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

    with torch.no_grad():
        ref_out = ref_model(*inputs)
        test_out = test_model(*inputs)

        def _unwrap(x):
            return x[0] if isinstance(x, (list, tuple)) else x
        ref_t, test_t = map(_unwrap, (ref_out, test_out))

        diff = (test_t - ref_t).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        passed = torch.allclose(ref_t, test_t, atol=tol, rtol=tol)

    return passed, max_err, mean_err





def benchmark_both(ref_model: torch.nn.Module,
                   test_model: torch.nn.Module,
                   device: torch.device,
                   warmup: int = 5,
                   repeat: int = 20) -> Tuple[float, float]:
    
    import importlib.util
    import inspect

    file_path = inspect.getfile(ref_model.__class__)
    spec = importlib.util.spec_from_file_location("ref_mod", file_path)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)

    get_inputs = getattr(ref_mod, "get_inputs", None)
    if get_inputs is None:
        raise RuntimeError("ref_model not define function get_inputs()")

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

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
        t1 = time.perf_counter()
        ref_times.append((t1 - t0) * 1000)

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
        t1 = time.perf_counter()
        test_times.append((t1 - t0) * 1000)

    ref_avg = sum(ref_times) / len(ref_times)
    test_avg = sum(test_times) / len(test_times)
    return ref_avg, test_avg



if __name__ == "__main__":
    PY = Path(r"D:\data\design\code\Kernel\agent\prompt\oneshot\model_new_ex_add.py")
    ok, log = compile_kernel(PY)
    print("编译结果:", "✅ 成功" if ok else "❌ 失败")
    if not ok:
        print("日志:", log)
    