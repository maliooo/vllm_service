import argparse
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import time
from rich import print
import os
import json
LORA_PATH = {
    "文生图": os.path.join(os.path.dirname(__file__), "lora_dir/checkpoint-1875-文生图"),
    # "文生图": "/home/zhangxuqi/malio/test/code/66/111/checkpoint-1875",
    # "图生图无反解": "./models/Qwen/Qwen2.5-0.5B-Instruct",
    # "图生图有反解": "./models/Qwen/Qwen2.5-0.5B-Instruct",
}
LORA_PATH_ID = {
    "文生图": 1,
    "图生图无反解": 2,
    "图生图有反解": 3,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用vLLM部署Qwen2.5-0.5B-Instruct模型服务")
    parser.add_argument("--model-path", type=str, default="models/Qwen2.5-0.5B-Instruct", 
                        help="模型路径")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                        help="张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3, 
                        help="GPU内存利用率")
    parser.add_argument("--max-model-len", type=int, default=512, 
                        help="最大模型长度")
    parser.add_argument("--enforce-eager", action="store_true", 
                        help="强制使用eager模式")
    parser.add_argument("--dtype", type=str, default="float16", 
                        choices=["float16", "bfloat16", "float32"],
                        help="模型数据类型")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="服务主机")
    parser.add_argument("--port", type=int, default=8000, 
                        help="服务端口")
    
    return parser.parse_args()

# 定义请求模型
class GenerationRequest(BaseModel):
    prompt: str  # 构建好的prompt，提示词应用号的前缀
    user_input: str  # 用户输入前缀
    user_id: int  # 用户id
    image_description: str = None  # 图片描述
    max_tokens: int = 64  # 最大token数
    temperature: float = 1.0  # 温度
    top_p: float = 0.9  # top_p
    top_k: int = 50  # top_k
    repetition_penalty: float = 1.1  # 重复惩罚
    stop_sequences: List[str] = Field(default_factory=list)  # 停止序列
    type: str = "文生图"  # 类型
    use_lora: bool = True  # 是否使用lora
    compute_tokens: bool = False  # 是否计算token, 默认不计算,计算会增加计算时间

# 定义响应模型
class GenerationResponse(BaseModel):
    generated_text: str
    token_cost: Dict[str, int]
    time_cost: str
    llm_template_prompt: str
    llm_outputs: str
    lora_request: Dict[str, str]
    error: Optional[str] = None


def load_model(args):
    """加载模型和分词器"""
    print(f"[blue]加载模型：{args.model_path}[/blue]")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 加载模型
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        # dtype=args.dtype,
        enable_lora=True,
    )
    
    print(f"模型 {args.model_path} 已成功加载")
    return llm, tokenizer


def build_prompt(prompt, type="文生图", space="", style="", description=""):
    # if type == "文生图":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过40字，只输出中文续写内容，不要多余的解释。"""
    # elif type == "图生图无反解":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}"""
    # elif type == "图生图有反解":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}\n底图内容：{description}"""
    
    # 文生图
    template_prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过40字，只输出中文续写内容，不要多余的解释。<|im_end|>\n<|im_start|>assistant\n{prompt}"
    return template_prompt

# 全局变量
# 创建FastAPI应用
app = FastAPI(title="Qwen2.5-0.5B-Instruct with LoRA API")
args = parse_args()
# 加载分词器
llm, tokenizer = load_model(args)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """生成文本的API端点"""
    st = time.time()
    print(f"[blue]获得用户输入：{request}[/blue]")
    print("-"*10)

    # 设置采样参数
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        stop=request.stop_sequences if request.stop_sequences else None,
    )
    
    # 准备LoRA请求
    if request.use_lora:
        lora_request = LoRARequest(
            lora_name=request.type,
            lora_int_id=LORA_PATH_ID[request.type],  # 内部ID，可以是任意整数
            lora_path=LORA_PATH[request.type]
        )
        print(f"[green]使用lora_name：{request.type}[/green]")
        print(f"[green]使用lora_int_id：{LORA_PATH_ID[request.type]}[/green]")
        print(f"[green]使用lora_path：{LORA_PATH[request.type]}[/green]")
    else:
        lora_request = None
    

    input_tokens = 0
    output_tokens = 0
    error = None
    template_prompt = ""
    generated_text = ""
    outputs = ""

    # 生成文本
    # global llm
    template_prompt = build_prompt(request.user_input, type=request.type)
    print(f"[yellow]使用prompt：{template_prompt}[/yellow]")
    try:
        outputs = llm.generate(
            [template_prompt], 
            sampling_params, 
            lora_request=lora_request if request.use_lora else None
        )
        # 获取生成的文本
        generated_text = outputs[0].outputs[0].text
        if request.compute_tokens:
            # 获取输入token数量
            input_tokens = len(tokenizer.encode(template_prompt))
            # 计算输出token数量
            output_tokens = len(tokenizer.encode(generated_text))

    except Exception as e:
        print(f"[red]生成失败: {e}[/red]")
        error = str(e)

    
    # 构建响应
    response = {
        "generated_text": generated_text,
        "token_cost": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        },
        "time_cost": f"{time.time() - st:.2f}秒",  # 函数运行时间
        "llm_template_prompt": template_prompt, # llm实际使用的prompt
        "llm_outputs": f"{outputs}", # llm的输出
        "lora_request": {
            "lora_name": str(lora_request.lora_name) if lora_request else "",
            "lora_int_id": str(lora_request.lora_int_id) if lora_request else "",
            "lora_path": str(lora_request.lora_path) if lora_request else "",
        },
        "error": error
    }
    # print(f"用户{request.user_id}的输入：{request.user_input}\n生成结果：{generated_text}\n耗时：{time.time() - st:.2f}秒\n\n")
    # print("-"*100)
    # print(f"返回结果: {response}")
    return response

# @app.post("/add_lora")
# async def add_lora(request: Request):
#     """添加LoRA适配器的API端点"""
#     data = await request.json()
#     lora_id = data.get("lora_id")
#     lora_path = data.get("lora_path")
    
#     if not lora_id or not lora_path:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "lora_id和lora_path是必需的"}
#         )
    
#     try:
#         llm.add_lora(lora_id, lora_path)
#         return {"status": "success", "message": f"LoRA {lora_id} 已成功添加"}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"添加LoRA时出错: {str(e)}"}
#         )

# @app.post("/remove_lora")
# async def remove_lora(request: Request):
#     """移除LoRA适配器的API端点"""
#     data = await request.json()
#     lora_id = data.get("lora_id")
    
#     if not lora_id:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "lora_id是必需的"}
#         )
    
#     try:
#         llm.remove_lora(lora_id)
#         return {"status": "success", "message": f"LoRA {lora_id} 已成功移除"}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"移除LoRA时出错: {str(e)}"}
#         )

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}



if __name__ == "__main__":
    import uvicorn

    # 启动服务
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        workers=1  # vLLM目前不支持多worker
    )
