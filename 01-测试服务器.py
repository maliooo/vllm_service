import requests
import json
import time
import argparse
from rich import print

LORA_PATH = {
    "文生图": "./models/Qwen/Qwen2.5-0.5B-Instruct",
    "图生图无反解": "./models/Qwen/Qwen2.5-0.5B-Instruct",
    "图生图有反解": "./models/Qwen/Qwen2.5-0.5B-Instruct",
}

def test_vllm_service(host="localhost", port=8000, lora_path=None, user_input=None):
    """测试vLLM服务的各项功能"""
    base_url = f"http://{host}:{port}"
    
    print("="*50)
    print("开始测试vLLM服务")
    print("="*50)
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ 健康检查通过")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 健康检查请求异常: {str(e)}")
        return
    
    # 测试基本生成功能
    print("\n测试基本文本生成...")
    generation_request = {
        "prompt": user_input,
        "user_input": user_input,
        "user_id": 1,
        "max_tokens": 32,
        "compute_tokens": True,
        "use_lora": True,
        # "temperature": 0.7,
        # "top_p": 0.9
    }
    
    try:
        st = time.time()
        response = requests.post(
            f"{base_url}/generate", 
            json=generation_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 基本生成成功, 用户输入: {user_input}")
            print(f"生成文本: {result['generated_text']}")
            print(f"Token统计: {result['token_cost']}")
            print(f"[blue]耗时: {(time.time() - st):.2f}秒[/blue]")
            print(f"返回结果: {result}")
        else:
            print(f"❌ 基本生成失败: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ 基本生成请求异常: {str(e)}")
        return
    
    # 如果提供了LoRA路径，测试LoRA功能
    if lora_path:
        # 测试添加LoRA
        print("\n测试添加LoRA...")
        lora_id = "test_lora"
        add_lora_request = {
            "lora_id": lora_id,
            "lora_path": lora_path
        }
        
        try:
            response = requests.post(
                f"{base_url}/add_lora", 
                json=add_lora_request
            )
            
            if response.status_code == 200:
                print(f"✅ 添加LoRA成功: {response.json()['message']}")
            else:
                print(f"❌ 添加LoRA失败: {response.status_code}")
                print(response.text)
                return
        except Exception as e:
            print(f"❌ 添加LoRA请求异常: {str(e)}")
            return
        
        # 测试使用LoRA生成
        print("\n测试使用LoRA生成...")
        lora_generation_request = {
            "prompt": "请用简短的语言介绍一下人工智能:",
            "lora_id": lora_id,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{base_url}/generate", 
                json=lora_generation_request
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ LoRA生成成功")
                print(f"生成文本: {result['generated_text']}")
                print(f"Token统计: {result['usage']}")
            else:
                print(f"❌ LoRA生成失败: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ LoRA生成请求异常: {str(e)}")
        
        # 测试移除LoRA
        print("\n测试移除LoRA...")
        remove_lora_request = {
            "lora_id": lora_id
        }
        
        try:
            response = requests.post(
                f"{base_url}/remove_lora", 
                json=remove_lora_request
            )
            
            if response.status_code == 200:
                print(f"✅ 移除LoRA成功: {response.json()['message']}")
            else:
                print(f"❌ 移除LoRA失败: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ 移除LoRA请求异常: {str(e)}")
    
    # print("\n="*50)
    print("测试完成")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试vLLM服务")
    parser.add_argument("--host", type=str, default="localhost", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA模型路径，用于测试LoRA功能")
    parser.add_argument("--user-input", type=str, default="自然光，现代客厅，木质家具，简约风格，客厅设计", help="用户输入")
    
    args = parser.parse_args()
    test_vllm_service(args.host, args.port, args.lora_path, args.user_input)

    # 测试图片生成
    # test_vllm_service(args.host, args.port, args.lora_path, image_description="一个美丽的女孩")