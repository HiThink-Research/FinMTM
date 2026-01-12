

from etest import *
# ------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct")
VLLM_MODEL_NAME = os.environ.get("CURRENT_VLM_MODEL", "Qwen3-VL-30B-A3B-Instruct")
# VLLM_MODEL_NAME = "/mnt/HithinkOmni/user_workspace/zhangchenxi4/checkpoints/qwen_model"

# 使用 '/' 分割，并取最后一个元素
VLLM_MODEL_NAME = VLLM_MODEL_NAME.split('/')[-1]
print(VLLM_MODEL_NAME)
# ======= 主目录：这里放多个 input 文件 =======
INPUT_DIR = "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ"
OUTPUT_DIR = os.path.join(INPUT_DIR, VLLM_MODEL_NAME)

PORT=9000
api_base = f"http://localhost:{PORT}/v1"
if __name__ == "__main__":
    input_jsonl = "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ/output.jsonl"
#评测单选题
    evaluate_jsonl_with_accuracy(
        input_jsonl=input_jsonl,
        output_jsonl=os.path.join(OUTPUT_DIR,"eval_results_s_nt.jsonl"),
        summary_json=os.path.join(OUTPUT_DIR,"eval_summary_s_nt.jsonl"),
        api_base=api_base,
        model=MODEL_NAME
    )
#评测多选题
    input_jsonl = "/cpfs01/HithinkOmniSSD/user_workspace/ganziliang/code/omini/OQ/output_wrong_multi.jsonl"
    evaluate_jsonl_with_accuracy(
        input_jsonl=input_jsonl,
        output_jsonl=os.path.join(OUTPUT_DIR,"eval_results_m.jsonl"),
        summary_json=os.path.join(OUTPUT_DIR,"eval_summary_m.jsonl"),
        api_base=api_base,
        model=MODEL_NAME,
    )
