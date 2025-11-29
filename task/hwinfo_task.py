import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import time
import torch
import traceback

from pathlib import Path
from typing import Dict, List, Tuple


from agent.llm import LLM
from utils.utils import correctness_and_benchmark, load_models_and_inputs, read_file, strip_fence, write_file, compile_kernel, benchmark_both, correctness_test
from agent.prompt.hwinfo_prompt import hwinfo_prompt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

async def run_hwinfo_task(
        tasks: List[Path],
        llm: LLM,
        run_dir: Path,
        gpu_name: str) -> None:

    llm.change_temperature(0.7)
    comp_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 16)


    jobs = [(task, gpu_name) for task in tasks]
    codes = await fetch_hwinfo_compile(jobs, llm, comp_pool, run_dir)

    for task in tasks:
        task_name = task.stem
        task_root = (run_dir / task.parent.name / task_name).resolve()
        code_file = codes[task]
        test_one_hw(task, code_file, run_dir)

        eval_file = code_file.parent.parent / "eval.json"
        summary = json.loads(eval_file.read_text())
        print(f"[{task_name}] speedup {summary['speedup']:.3f}x  -> {eval_file}")

# --------------------------------------------------

async def fetch_hwinfo_compile(
        jobs: List[Tuple[Path, str]], llm: LLM, pool, run_dir: Path
) -> Dict[Path, Path]:
    llm_sem = asyncio.Semaphore(20)
    loop    = asyncio.get_running_loop()

    async def fetch(job: Tuple[Path, str]):
        task, gpu = job
        async with llm_sem:
            return await llm_achat_write_hw(task, gpu, llm, run_dir)

    llm_futs = [asyncio.create_task(fetch(j)) for j in jobs]
    code_files = await asyncio.gather(*llm_futs)

    compile_futs = [loop.run_in_executor(pool, compile_one_hw, task, cf)
                    for (task, _), cf in zip(jobs, code_files)]
    await asyncio.gather(*compile_futs)

    return dict(zip([t for t, _ in jobs], code_files))

# --------------------------------------------------

async def llm_achat_write_hw(
        task: Path, gpu_name: str, llm: LLM, run_dir: Path) -> Path:
    task_name = task.stem
    task_root = (run_dir / task.parent.name / task_name).resolve()
    task_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task, task_root / "ref.py")

    rnd_dir = task_root / "round_000"
    (rnd_dir / "code").mkdir(parents=True, exist_ok=True)
    (rnd_dir / "llm_io").mkdir(parents=True, exist_ok=True)

    arch_src = read_file(task)
    prompt   = hwinfo_prompt(arch_src, gpu_name)
    code     = strip_fence(await llm.achat(prompt))

    code_file = rnd_dir / "code" / "generated.py"
    write_file(str(code_file), code)
    (rnd_dir / "llm_io" / "prompt.txt").write_text(prompt, encoding="utf-8")
    (rnd_dir / "llm_io" / "reply.txt").write_text(code, encoding="utf-8")
    return code_file

def compile_one_hw(task: Path, code_file: Path) -> None:
    ok, log = compile_kernel(code_file)
    eval_file = code_file.parent.parent / "eval.json"
    if not ok:
        summary = {"compiled": False, "error": "compile", "error_log": log}
        eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"[{task.stem}] compile failed")

# --------------------------------------------------

def test_one_hw(task: Path, code_file: Path, run_dir: Path) -> None:
    task_root = code_file.parent.parent
    eval_file = task_root / "eval.json"
    summary = {
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
        return

    try:
        passed, max_err, mean_err, ref_avg, test_avg = correctness_and_benchmark(
            RefModel, ModelNew, inputs, DEVICE, tol=1e-3, warmup=5, repeat=10)
        summary.update(
            correct=passed, max_err=max_err, mean_err=mean_err,
            ref_avg_ms=ref_avg, test_avg_ms=test_avg,
            speedup=ref_avg / test_avg if ref_avg else 0.0)
    except Exception as e:
        summary.update(error="correctness_and_benchmark", error_log=traceback.format_exc())

    eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")


    