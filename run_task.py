import os
import argparse
import logging
import shutil

from agent.llm import LLM
from pathlib import Path
from typing import List
from datetime import datetime


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    parser.add_argument("--server_name", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--task_level", type=int, default=0, choices=[0, 1, 2, 3, 4], help="task level")
    parser.add_argument("--task_id", type=int, default=0, help="task id")
    parser.add_argument("--task_file", type=str, default="./KernelBench")
    parser.add_argument("--task_type",type=str,choices=["oneshot", "sampling", "refine", "hwinfo"],default="oneshot",help="type of task to run")
    parser.add_argument("--gpu_name", type=str, default="A100", help="GPU name for hwinfo task")
    return parser


def collect_task(root: Path, task_level = 0, task_id = 0):
    tasks = []
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    levels = range(1, 4) if task_level == 0 else [task_level]

    tasks = []
    for lvl in levels:
        lvl_dir = root / f"level{lvl}"
        if not lvl_dir.is_dir():
            continue
        py_files = sorted(lvl_dir.glob("*.py"), key=lambda p: int(p.stem.split('_')[0]))
        if task_id == 0:          
            tasks.extend(py_files)
        else:                     
            if 1 <= task_id <= len(py_files):
                tasks.append(py_files[task_id - 1])
    return tasks
    
def make_run_dir(base_dir: Path, server_name: str, model: str, task_type: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{server_name}_{model}_{task_type}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run directory created at: {run_dir}")
    return run_dir

def main():
    arg = build_parser().parse_args()

    tasks = collect_task(Path(arg.task_file), task_level=arg.task_level, task_id=arg.task_id)

    dir = make_run_dir(Path("./runs"), arg.server_name, arg.model, arg.task_type)

    llm = LLM(server_name=arg.server_name, model=arg.model, max_tokens=arg.max_tokens, temperature=arg.temperature, top_p=arg.top_p)

    task_type = arg.task_type

    if task_type == "oneshot":
        from task.oneshot_task import run_oneshot_task
        run_oneshot_task(tasks, llm, dir)

    elif task_type == "sampling":
        from task.sampling_task import run_sampling_task
        run_sampling_task(tasks, llm, dir)

    elif task_type == "refine":
        from task.refine_task import run_refine_task
        run_refine_task(tasks, llm, dir)

    elif task_type == "hwinfo":
        from task.hwinfo_task import run_hwinfo_task
        run_hwinfo_task(llm, tasks, dir, arg.gpu_name)

    
if __name__ == "__main__":
    main()
     





#def main():
#    arg = build_parser().parse_args()
#
#    tasks = collect_task(Path(arg.task_file), task_level=0, task_id=0)
#
#    dir = make_run_dir(Path("./runs"), arg.server_name, arg.model, arg.task_type)
#
#    llm = LLM(server_name=arg.server_name, model=arg.model, max_tokens=arg.max_tokens, temperature=arg.temperature, top_p=arg.top_p)

    