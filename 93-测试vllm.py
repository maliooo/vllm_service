import torch
from vllm import LLM
from vllm.lora.request import LoRARequest
from rich import print
import os
import time
import pandas as pd  # pip install pandas

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def build_conversion_prompt(prompt, type="文生图", space="", style="", description=""):

    if type == "文生图":
        prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过40字，只输出中文续写内容，不要多余的解释。"""
    elif type == "图生图无反解":
        prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}"""
    elif type == "图生图有反解":
        prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}\n底图内容：{description}"""
            
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt
        },
    ]
    return conversation

def build_prompt(prompt, type="文生图", space="", style="", description=""):
    # if type == "文生图":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过40字，只输出中文续写内容，不要多余的解释。"""
    # elif type == "图生图无反解":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}"""
    # elif type == "图生图有反解":
    #     prompt = f"""作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，结合如下空间和风格场景，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过30字，只输出中文续写内容，不要多余的解释。\n空间：{space}\n风格场景：{style}\n底图内容：{description}"""
    
    template_prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n作为stable diffusion XL的提示词工程师，请对“{prompt}”进行续写。要求从专业的室内/建筑/景观领域和多模态生成模型提示词生效能力考虑，生成更适配stable diffusion XL模型风格的续写词。续写结果不超过40字，只输出中文续写内容，不要多余的解释。<|im_end|>\n<|im_start|>assistant\n{prompt}"
    return template_prompt



if __name__ == "__main__":

    # vllm 加载本地qweb模型
    model_path = r"/home/public/ai_chat_data/models/Qwen/Qwen2.5-0.5B-Instruct"
    lora_path = r"/home/zhangxuqi/malio/test/code/66/111/checkpoint-1875"
    csv_path = r"/home/zhangxuqi/malio/test/code/66/111/大模型场景验收 - 提示词补全 (2).csv"
    output_csv_path = r"/home/zhangxuqi/malio/test/code/66/111/大模型场景验收-提示词补全-vllm.csv"
    df = pd.read_csv(csv_path)
    # 前缀不为空
    df = df[df["前缀"].notna()]
    print(df.shape)
    # 设置显卡为6卡
    # exit()


    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        # max_model_len=1024,
        enable_lora=True,
        tensor_parallel_size=1,  # 减少并行大小
        gpu_memory_utilization=0.3,  # 控制GPU内存使用率
        # max_num_batched_tokens=4096,  # 限制批处理的最大token数
        max_num_seqs=512,  # 限制序列数量
        # swap_space=4 * 1024 * 1024 * 1024,  # 启用4GB的CPU交换空间
        # 使用cpu推理
    )


    print("开始进行......")
    for index, row in df.iterrows():
        prompt = row["前缀"]
        # conversation = build_conversion_prompt(prompt)
        template_prompt = build_prompt(prompt)
        t1 = time.time()
        result = llm.generate(
                        template_prompt,
                        lora_request=LoRARequest("sql_adapter", 1, lora_path)
                    )
        # 只取vllm的输出
        generate_text = result[0].outputs[0].text
        t2 = time.time()
        print(f"前缀：[red]{prompt}[/red]\n续写：[green]{generate_text}[/green]\n耗时：[blue]{t2-t1}[/blue]秒")
        print("-"*100)

        # 没有lora
        t1 = time.time()
        result = llm.generate(
                        template_prompt,
                        # lora_request=LoRARequest("sql_adapter", 1, lora_path)
                    )
        # 只取vllm的输出
        generate_text = result[0].outputs[0].text
        t2 = time.time()
        print(f"\n\n没有lora\n前缀：[red]{prompt}[/red]\n续写：[green]{generate_text}[/green]\n耗时：[blue]{t2-t1}[/blue]秒")
        print("-"*100)

        # 保存结果
        model_name = model_path.split("/")[-1]
        df.loc[index, "前缀提示"] = prompt
        df.loc[index, f"{model_name}_vllm_prompt"] = template_prompt
        df.loc[index, f"{model_name}_vllm"] = generate_text
        df.loc[index, f"{model_name}_耗时"] = t2-t1

    # 保存结果
    # df.to_csv(output_csv_path, index=False)
    template_prompt = build_prompt(prompt)
    result = llm.generate(
                template_prompt,
                lora_request=LoRARequest("sql_adapter", 1, lora_path)
            )








