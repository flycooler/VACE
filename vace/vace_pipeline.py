import torch
import argparse
import importlib
from typing import Dict, Any

def load_parser(module_name: str) -> argparse.ArgumentParser:
    # English: Dynamically loads a module by its name and returns its argument parser.
    # Chinese: 根据模块名称动态加载模块，并返回其参数解析器。
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_parser"):
        raise ValueError(f"{module_name} undefined get_parser()")
    return module.get_parser()

def filter_args(args: Dict[str, Any], parser: argparse.ArgumentParser) -> Dict[str, Any]:
    # English: Filters a dictionary of arguments, keeping only those defined in the provided parser.
    # Chinese: 过滤一个参数字典，只保留在给定的解析器中定义的参数。
    known_args = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            known_args.add(action.dest)
    return {k: v for k, v in args.items() if k in known_args}

def main():
    # English: Main parser to determine the pipeline base (e.g., 'ltx' or 'wan').
    # Chinese: 主解析器，用于确定要运行的流水线基础（例如 'ltx' 或 'wan'）。
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--base", type=str, default='ltx', choices=['ltx', 'wan'])
    pipeline_args, _ = main_parser.parse_known_args()

    # English: Select the appropriate preprocessing and inference modules based on the chosen base.
    # Chinese: 根据选择的基础模型，确定相应的预处理和推理模块。
    if pipeline_args.base in ["ltx"]:
        preproccess_name, inference_name = "vace_preproccess", "vace_ltx_inference"
    else:
        preproccess_name, inference_name = "vace_preproccess", "vace_wan_inference"

    # English: Load the argument parsers from both the preprocessing and inference modules.
    # Chinese: 从预处理和推理模块中加载各自的参数解析器。
    preprocess_parser = load_parser(preproccess_name)
    inference_parser = load_parser(inference_name)

    # English: Merge all arguments from the sub-parsers into the main parser.
    #          This allows users to provide all arguments in a single command line.
    # Chinese: 将所有子解析器的参数合并到主解析器中。
    #          这允许用户在单个命令行中提供所有参数。
    for parser in [preprocess_parser, inference_parser]:
        for action in parser._actions:
            if action.dest != "help":
                main_parser._add_action(action)

    # English: Parse all command-line arguments using the combined parser.
    # Chinese: 使用合并后的解析器解析所有命令行参数。
    cli_args = main_parser.parse_args()
    args_dict = vars(cli_args)

    # --- Step 1: Run Preprocessing ---
    # --- 步骤 1：运行预处理 ---
    # English: Filter arguments relevant to the preprocessor and run its main function.
    # Chinese: 筛选出与预处理器相关的参数，并运行其 main 函数。
    preprocess_args = filter_args(args_dict, preprocess_parser)
    preprocesser = importlib.import_module(preproccess_name)
    preprocess_output = preprocesser.main(preprocess_args)
    print("preprocess_output:", preprocess_output)

    # English: Clean up memory to prepare for the inference step.
    # Chinese: 清理内存，为推理步骤做准备。
    del preprocesser
    torch.cuda.empty_cache()

    # --- Step 2: Run Inference ---
    # --- 步骤 2：运行推理 ---
    # English: Filter arguments for the inference module.
    # Chinese: 筛选出用于推理模块的参数。
    inference_args = filter_args(args_dict, inference_parser)
    # English: CRITICAL STEP: Update inference arguments with the output from the preprocessing step.
    #          This is how the two stages are connected.
    # Chinese: 关键步骤：使用预处理步骤的输出更新推理参数。
    #          这是连接两个阶段的方式。
    inference_args.update(preprocess_output)
    # English: Run the main inference function with the combined arguments.
    # Chinese: 使用合并后的参数运行主推理函数。
    inference_output = importlib.import_module(inference_name).main(inference_args)
    print("inference_output:", inference_output)


if __name__ == "__main__":
    # English: Entry point of the script.
    # Chinese: 脚本的入口点。
    main()