import json
import shutil
import time
import torch
import traceback

from pathlib import Path
from typing import List


from agent.llm import LLM
from utils.utils import read_file, write_file, compile_kernel, benchmark_both, correctness_test
from agent.prompt.oneshot_prompt import oneshot_prompt
from agent.prompt.refine_prompt import refine_prompt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_refine_task(tasks: List[Path],
                    llm: LLM,
                    run_dir: Path,
                    epoch_per_task: int = 10,
                    warmup: int = 5,
                    repeat: int = 20,
                    tol: float = 1e-3) -> None:

    for task in tasks:
        task_name = task.stem
        task_root = (run_dir / task.parent.name / task_name).resolve()
        task_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task, task_root / "ref.py")

        ref_src = read_file(task) 
        best_speedup = 0.0
        best_epoch = -1
        epoch_meta = []

        for e in range(epoch_per_task):
            epoch_dir = task_root / f"epoch_{e:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "code").mkdir(parents=True, exist_ok=True)
            (epoch_dir / "llm_io").mkdir(parents=True, exist_ok=True)

            code_file = epoch_dir / "code" / "generated.py"
            eval_file = epoch_dir / "eval.json"

            
            if e == 0:
                prompt = oneshot_prompt(ref_src)
                feedback, runtime = "No previous iteration.", 0.0
            else:
                
                last_eval = json.loads((task_root / f"epoch_{e-1:03d}" / "eval.json").read_text())
                feedback = last_eval.get("feedback", "No feedback from previous iteration.")
                runtime = last_eval.get("ref_avg_ms", 0.0)
                latest_src = read_file(task_root / f"epoch_{e-1:03d}" / "code" / "generated.py")
                prompt = refine_prompt(ref_src, latest_src, feedback, runtime)

            output = llm.chat(prompt)
            write_file(str(code_file), output)
            (epoch_dir / "llm_io" / "prompt.txt").write_text(prompt, encoding="utf-8")
            (epoch_dir / "llm_io" / "reply.txt").write_text(output, encoding="utf-8")

            
            summary = {
                "epoch": e,
                "compiled": False,
                "correct": False,
                "speedup": 0.0,
                "ref_avg_ms": 0.0,
                "test_avg_ms": 0.0,
                "max_err": 0.0,
                "error": None,
                "error_log": None,
            }

            try:
                
                ok, build_log = compile_kernel(code_file)
                if not ok:
                    summary["error"] = "compile"
                    summary["error_log"] = build_log
                    raise Exception("Compilation failed")

                
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"mod_e{e}", code_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                RefModel = getattr(mod, "Model", None)
                ModelNew = getattr(mod, "ModelNew", None)
                if None in (RefModel, ModelNew):
                    raise RuntimeError("Model or ModelNew missing")

               
                passed, max_err, _ = correctness_test(RefModel(), ModelNew(), DEVICE, tol=tol)
                summary["correct"] = passed
                summary["max_err"] = max_err

                
                ref_avg, test_avg = benchmark_both(RefModel(), ModelNew(), DEVICE, warmup=warmup, repeat=repeat)
                summary["ref_avg_ms"] = ref_avg
                summary["test_avg_ms"] = test_avg
                summary["speedup"] = ref_avg / test_avg if ref_avg else 0.0

                
                if summary["correct"] and summary["speedup"] > best_speedup:
                    best_speedup = summary["speedup"]
                    best_epoch = e

            except Exception as ex:
                summary["error"] = "runtime"
                summary["error_log"] = traceback.format_exc()

            finally:
                
                eval_file.write_text(json.dumps(summary), encoding="utf-8")
                print(f"[{task_name}][e{e:03d}] speedup {summary['speedup']:.3f}x")

            
            epoch_meta.append(summary)

        task_meta = {
            "best_epoch": best_epoch,
            "best_speedup": best_speedup,
            "epochs": epoch_meta,
        }
        (task_root / "epoch_meta.json").write_text(json.dumps(task_meta), encoding="utf-8")
        print(f"[{task_name}] best epoch {best_epoch}  speedup {best_speedup:.3f}x")


        

        
