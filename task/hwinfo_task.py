import json
import shutil
import time
import torch
import traceback

from pathlib import Path
from typing import List


from agent.llm import LLM
from utils.utils import read_file, write_file, compile_kernel, benchmark_both, correctness_test
from agent.prompt.hwinfo_prompt import hwinfo_prompt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_oneshot_task(tasks: List[Path], llm: LLM, run_dir: Path, gpu_name: str) -> None:
    for task in tasks:
        task_name = task.stem
        task_root = (run_dir / task.parent.name / task_name).resolve()
        task_root.mkdir(parents=True, exist_ok=True)

        shutil.copy2(task, task_root / "ref.py")

        rnd_dir = task_root / "round_000"
        (rnd_dir / "code").mkdir(parents=True, exist_ok=True)
        (rnd_dir / "llm_io").mkdir(parents=True, exist_ok=True)

        code_file = rnd_dir / "code" / "generated.py"
        arch_src = read_file(task)
        prompt = hwinfo_prompt(arch_src, gpu_name)
        output = llm.chat(prompt)
        write_file(str(code_file), output)


        (rnd_dir / "llm_io" / "prompt.txt").write_text(prompt, encoding="utf-8")
        (rnd_dir / "llm_io" / "reply.txt").write_text(output, encoding="utf-8")

        eval_file = rnd_dir / "eval.json"
        summary = {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "ref_avg_ms": 0.0,
            "test_avg_ms": 0.0,
            "max_err": 0.0,
            "error": None,
            "error_log": None,
        }

        ok, build_log = compile_kernel(code_file)
        if not ok:
            summary["error"] = "compile"
            summary["error_log"] = build_log
            eval_file.write_text(json.dumps(summary), encoding="utf-8")
            continue
        summary["compiled"] = True

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("gen_mod", code_file)
            gen_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_mod)
            RefModel = getattr(gen_mod, "Model", None)
            ModelNew = getattr(gen_mod, "ModelNew", None)
            if None in (RefModel, ModelNew):
                raise RuntimeError("Model or ModelNew missing")
        except Exception as e:
            summary["error"] = "import"
            summary["error_log"] = traceback.format_exc()
            eval_file.write_text(json.dumps(summary), encoding="utf-8")
            continue

        try:
            passed, max_err, _ = correctness_test(RefModel(), ModelNew(), DEVICE)
            summary["correct"] = passed
            summary["max_err"] = max_err
        except Exception as e:
            summary["error"] = "correctness"
            summary["error_log"] = traceback.format_exc()
            eval_file.write_text(json.dumps(summary), encoding="utf-8")
            continue

        try:
            ref_avg, test_avg = benchmark_both(RefModel(), ModelNew(), DEVICE, warmup=5, repeat=10)
            summary["ref_avg_ms"] = ref_avg
            summary["test_avg_ms"] = test_avg
            summary["speedup"] = ref_avg / test_avg if ref_avg else 0.0
        except Exception as e:
            summary["error"] = "benchmark"
            summary["error_log"] = traceback.format_exc()
            eval_file.write_text(json.dumps(summary), encoding="utf-8")
            continue

        eval_file.write_text(json.dumps(summary), encoding="utf-8")
        print(f"[{task_name}] speedup {summary['speedup']:.3f}x  -> {eval_file}")
        


    