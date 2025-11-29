
import json
import shutil
import time
import torch
import traceback
import asyncio
import os


from pathlib import Path
from typing import List, Dict
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from agent.llm import LLM
from utils.utils import correctness_and_benchmark, load_models_and_inputs, read_file, strip_fence, write_file, compile_kernel, benchmark_both, correctness_test
from agent.prompt.oneshot_prompt import oneshot_prompt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import asyncio
import os
from pathlib import Path
from typing import List, Dict

# --------------------------------------------------

async def run_sampling_task(
        tasks: List[Path],
        llm: LLM,
        run_dir: Path,
        *,
        samples_per_task: int = 4,
        warmup: int = 5,
        repeat: int = 20,
        tol: float = 1e-3) -> None:

    llm.change_temperature(1.6)

    all_jobs = [(task, b) for task in tasks for b in range(samples_per_task)]
    codes = await fetch_and_compile_sub(all_jobs, llm, run_dir)

    from collections import defaultdict
    task_groups = defaultdict(list)
    for (task, b), code_file in codes.items():
        task_groups[task].append((b, code_file))

    for task, branches in task_groups.items():
        task_name = task.stem
        task_root = (run_dir / task.parent.name / task_name).resolve()
        branch_meta = []
        for b, code_file in branches:
            test_one(task, code_file, run_dir)  
            eval_file = code_file.parent.parent / "eval.json"
            branch_meta.append(json.loads(eval_file.read_text()))

        (task_root / "branch_meta.json").write_text(
            json.dumps(branch_meta, indent=4, ensure_ascii=False), encoding="utf-8")
        best = max(branch_meta, key=lambda x: x.get("speedup", 0))
        print(f"[{task_name}] best branch {best['branch']:03d}  speedup {best['speedup']:.3f}x")

# --------------------------------------------------

def expand_branches(task: Path, task_root: Path, n: int) -> List[Tuple[Path, int]]:
    return [(task, b) for b in range(n)]

# --------------------------------------------------

async def fetch_and_compile_sub(
        sub_tasks: List[Tuple[Path, int]], llm: LLM , run_dir: Path) -> Dict[Tuple[Path, int], Path]:
    llm_sem   = asyncio.Semaphore(20)
    comp_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 16)
    loop = asyncio.get_running_loop()

    async def fetch(item: Tuple[Path, int]):
        task, b = item
        async with llm_sem:
            return await llm_achat_write_b(task, b, llm, run_dir)

    llm_futs = [asyncio.create_task(fetch(it)) for it in sub_tasks]
    code_files = await asyncio.gather(*llm_futs)

    compile_futs = [loop.run_in_executor(comp_pool, compile_one_b, item, cf)
                    for item, cf in zip(sub_tasks, code_files)]
    await asyncio.gather(*compile_futs)

    return dict(zip(sub_tasks, code_files))

# --------------------------------------------------

async def llm_achat_write_b(
        task: Path,
        branch_id: int,
        llm: LLM,
        run_dir: Path) -> Path:         
    task_name = task.stem
    task_root = (run_dir / task.parent.name / task_name).resolve()
    task_root.mkdir(parents=True, exist_ok=True)

    branch_dir = task_root / f"branch_{branch_id:03d}"
    (branch_dir / "code").mkdir(parents=True, exist_ok=True)
    (branch_dir / "llm_io").mkdir(parents=True, exist_ok=True)

    arch_src = read_file(task)
    prompt   = oneshot_prompt(arch_src)
    code     = strip_fence(await llm.achat(prompt))

    code_file = branch_dir / "code" / "generated.py"
    write_file(str(code_file), code)
    (branch_dir / "llm_io" / "prompt.txt").write_text(prompt, encoding="utf-8")
    (branch_dir / "llm_io" / "reply.txt").write_text(code, encoding="utf-8")
    return code_file

def compile_one_b(item: Tuple[Path, int], code_file: Path) -> None:
    task, _ = item
    ok, log = compile_kernel(code_file)
    if not ok:
        eval_file = code_file.parent.parent / "eval.json"
        summary = {"branch": int(code_file.parent.parent.name.split("_")[1]),
                   "compiled": False, "error": "compile", "error_log": log}
        eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"[{task.stem}] compile failed")


def test_one(task: Path, code_file: Path, run_dir: Path) -> None:
    task_root = code_file.parent.parent
    eval_file = task_root / "eval.json"
    summary = {"compiled": True, "correct": False, "speedup": 0.0,
               "ref_avg_ms": 0.0, "test_avg_ms": 0.0,
               "max_err": 0.0, "mean_err": 0.0,
               "error": None, "error_log": None}

    try:
        RefModel, ModelNew, inputs = load_models_and_inputs(
            ref_py=task_root.parent / "ref.py", generated_py=code_file)
        if None in (RefModel, ModelNew):
            raise RuntimeError("Model or ModelNew missing")
    except Exception as e:
        summary.update(error="import", error_log=traceback.format_exc())
        eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
        return

    try:
        passed, max_err, mean_err, ref_avg, test_avg = correctness_and_benchmark(
            RefModel, ModelNew, inputs, DEVICE, tol=1e-3, warmup=5, repeat=10)
        summary.update(correct=passed, max_err=max_err, mean_err=mean_err,
                       ref_avg_ms=ref_avg, test_avg_ms=test_avg,
                       speedup=ref_avg/test_avg if ref_avg else 0.0)
    except Exception as e:
        summary.update(error="correctness_and_benchmark", error_log=traceback.format_exc())

    eval_file.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[{task.stem}] speedup {summary['speedup']:.3f}x  -> {eval_file}")