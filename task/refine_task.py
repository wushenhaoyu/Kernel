import asyncio
import json
import os
import shutil
import time
import torch
import traceback

from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from agent.llm import LLM
from utils.utils import correctness_and_benchmark, load_models_and_inputs, read_file, strip_fence, write_file, compile_kernel, benchmark_both, correctness_test
from agent.prompt.oneshot_prompt import oneshot_prompt
from agent.prompt.refine_prompt import refine_prompt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


async def run_refine_task(
        tasks: List[Path],
        llm: LLM,
        run_dir: Path,
        *,
        epoch_per_task: int = 10,
        warmup: int = 5,
        repeat: int = 20,
        tol: float = 1e-3) -> None:

    llm.change_temperature(0.7)
    comp_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 16)

    for epoch in range(epoch_per_task):
        print(f"========== Epoch {epoch:03d} ==========")


        epoch_jobs = [(task, epoch) for task in tasks]
        codes = await fetch_epoch_compile(epoch_jobs, llm, comp_pool, run_dir, epoch)

        for task in tasks:
            task_root = (run_dir / task.parent.name / task.stem).resolve()
            code_file = codes[(task, epoch)]
            summary   = test_one_epoch(task, epoch, code_file, task_root,
                                       tol, warmup, repeat)
            if summary["correct"] and summary["speedup"] > getattr(task_root, "_best", 0):
                task_root._best     = summary["speedup"]
                task_root._best_e   = epoch

        if epoch == epoch_per_task - 1:
            for task in tasks:
                task_name = task.stem
                task_root = (run_dir / task.parent.name / task_name).resolve()
                epoch_meta = [json.loads((task_root / f"epoch_{e:03d}" / "eval.json").read_text())
                              for e in range(epoch_per_task)]
                task_meta  = {
                    "best_epoch": getattr(task_root, "_best_e", -1),
                    "best_speedup": getattr(task_root, "_best", 0),
                    "epochs": epoch_meta
                }
                (task_root / "epoch_meta.json").write_text(
                    json.dumps(task_meta, indent=4, ensure_ascii=False), encoding="utf-8")
                print(f"[{task_name}] best epoch {task_meta['best_epoch']}  "
                      f"speedup {task_meta['best_speedup']:.3f}x")

# --------------------------------------------------

async def fetch_epoch_compile(
        epoch_jobs: List[Tuple[Path, int]], llm: LLM, pool, run_dir: Path, epoch: int
) -> Dict[Tuple[Path, int], Path]:
    llm_sem = asyncio.Semaphore(20)
    loop    = asyncio.get_running_loop()

    async def fetch(job: Tuple[Path, int]):
        task, _ = job
        async with llm_sem:
            return await llm_achat_write_e(task, epoch, llm, run_dir)

    llm_futs = [asyncio.create_task(fetch(j)) for j in epoch_jobs]
    code_files = await asyncio.gather(*llm_futs)

    compile_futs = [loop.run_in_executor(pool, compile_one_e, job, cf)
                    for job, cf in zip(epoch_jobs, code_files)]
    await asyncio.gather(*compile_futs)

    return dict(zip(epoch_jobs, code_files))

# --------------------------------------------------

async def llm_achat_write_e(
        task: Path, epoch: int, llm: LLM, run_dir: Path) -> Path:
    task_name = task.stem
    task_root = (run_dir / task.parent.name / task_name).resolve()
    task_root.mkdir(parents=True, exist_ok=True)
    if epoch == 0:                      
        shutil.copy2(task, task_root / "ref.py")

    epoch_dir = task_root / f"epoch_{epoch:03d}"
    (epoch_dir / "code").mkdir(parents=True, exist_ok=True)
    (epoch_dir / "llm_io").mkdir(parents=True, exist_ok=True)

    ref_src = read_file(task_root / "ref.py")
    if epoch == 0:
        prompt = oneshot_prompt(ref_src)
    else:
        last_eval = json.loads((task_root / f"epoch_{epoch-1:03d}" / "eval.json").read_text())
        feedback = last_eval.get("feedback", "No feedback from previous iteration.")
        runtime  = last_eval.get("ref_avg_ms", 0.0)
        latest_src = read_file(task_root / f"epoch_{epoch-1:03d}" / "code" / "generated.py")
        prompt = refine_prompt(ref_src, latest_src, feedback, runtime)

    code = strip_fence(await llm.achat(prompt))
    code_file = epoch_dir / "code" / "generated.py"
    write_file(str(code_file), code)
    (epoch_dir / "llm_io" / "prompt.txt").write_text(prompt, encoding="utf-8")
    (epoch_dir / "llm_io" / "reply.txt").write_text(code, encoding="utf-8")
    return code_file

def compile_one_e(job: Tuple[Path, int], code_file: Path) -> None:
    task, epoch = job
    ok, log = compile_kernel(code_file)
    eval_file = code_file.parent.parent / "eval.json"
    if not ok:
        summary = {"epoch": epoch, "compiled": False, "error": "compile", "error_log": log}
        eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"[{task.stem}][e{epoch:03d}] compile failed")

# --------------------------------------------------

def test_one_epoch(
        task: Path, epoch: int, code_file: Path, task_root: Path,
        tol: float, warmup: int, repeat: int) -> Dict:
    eval_file = task_root / f"epoch_{epoch:03d}" / "eval.json"
    summary = {
        "epoch": epoch,                     
        "compiled": True,
        "correct": False,
        "speedup": 0.0,
        "ref_avg_ms": 0.0,
        "test_avg_ms": 0.0,
        "max_err": 0.0,
        "mean_err": 0.0,
        "error": None,
        "error_log": None,
    }

    try:
        RefModel, ModelNew, inputs = load_models_and_inputs(
            ref_py=task_root.parent / "ref.py",
            generated_py=code_file)
        if None in (RefModel, ModelNew):
            raise RuntimeError("Model or ModelNew missing")
    except Exception as e:
        summary.update(error="import", error_log=traceback.format_exc())
        eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
        return summary

    try:
        passed, max_err, mean_err, ref_avg, test_avg = correctness_and_benchmark(
            RefModel, ModelNew, inputs, DEVICE, tol=tol, warmup=warmup, repeat=repeat)
        summary.update(
            correct=passed, max_err=max_err, mean_err=mean_err,
            ref_avg_ms=ref_avg, test_avg_ms=test_avg,
            speedup=ref_avg / test_avg if ref_avg else 0.0)
    except Exception as e:
        summary.update(error="correctness_and_benchmark", error_log=traceback.format_exc())

    eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[{task.stem}][e{epoch:03d}] speedup {summary['speedup']:.3f}x  -> {eval_file}")
    return summary

        
