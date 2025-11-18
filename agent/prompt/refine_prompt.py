import os

from utils.utils import read_file

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
    )
)

PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

def prompt_generate(
        ref_arch_src: str,
        latest_arch_src:str,
        feedback: str,
        runtime:float,
    ) -> str:
    prompt = PROBLEM_STATEMENT
    prompt += f"""Here is your reference architecture Model :\n
    ```
    {ref_arch_src}
    ``` \n
    """
    prompt += f"""Here is your latest generation:\n
    ```
    {latest_arch_src}
    ``` \n
    Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model . \n
    Here is your Evaluation Result: \n
    ``` 
    {feedback}
    ``` \n
    Your kernel executed successfully and produced the correct output . \n
    Here is your wall clock time : { runtime } milliseconds \n
    Name your new improved output architecture ModelNew . Output the new code in codeblocks . Please generate real code , NOT pseudocode , make sure the code compiles and is fully functional . Just output the new model code , no other text , and NO testing code !
    """
    return prompt

def refine_prompt(
        ref_arch_src: str,
        latest_arch_src:str,
        feedback: str,
        runtime:float,
    ) -> str:
    return prompt_generate(ref_arch_src, latest_arch_src, feedback, runtime)




