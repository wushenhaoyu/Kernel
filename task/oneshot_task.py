import shutil
import time



from pathlib import Path
from typing import List
from agent.llm import LLM
from utils.utils import read_file, write_file
from agent.prompt.oneshot_prompt import oneshot_prompt
def run_oneshot_task(tasks: List[Path], llm: LLM, run_dir: Path) -> None:

    for task in tasks:

        level_dir = task.parent          
        level_name = level_dir.name      
        task_name = task.stem            

        task_root = (run_dir / level_name / task_name).resolve()
        task_root.mkdir(parents=True, exist_ok=True)

        code_dir = task_root / "code"          
        io_dir = task_root / "llm_io"          
        log_dir = task_root / "logs"           
        #fig_dir = task_root / "figures"        
        #eval_dir = task_root / "evaluation"    

        for d in (code_dir, io_dir, log_dir):
            d.mkdir(exist_ok=True)

        shutil.copy2(task, task_root / "ref.py")

        arch_src = read_file(task)

        prompt = oneshot_prompt(arch_src)

        output = llm.chat(prompt)

        code_file = code_dir / f"generated_000.py"    
        write_file(str(code_file), output)

    