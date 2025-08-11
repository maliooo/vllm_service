#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLLM服务监控脚本
定时检查VLLM服务状态，如果服务不可用则自动重启
"""

import requests
import time
import os
import subprocess
import logging
import signal
import sys
from datetime import datetime
from utils import send_error_msg_to_feishu

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vllm_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# 配置
VLLM_HOST = "127.0.0.1"  # 服务器主机
VLLM_PORT = 28000        # 服务器端口
CHECK_INTERVAL = 5 * 60  # 检查间隔（秒）
RESTART_SCRIPT = f"{os.path.dirname(os.path.abspath(__file__))}/run.sh"  # 重启脚本路径
MAX_RESTART_ATTEMPTS = 3  # 最大重启尝试次数
RESTART_COOLDOWN = 60     # 重启冷却时间（秒）

# 请求超时时间（秒）
REQUEST_TIMEOUT = 10

def check_health():
    """检查服务健康状态"""
    try:
        # 先检查健康端点
        health_url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
        health_response = requests.get(health_url, timeout=REQUEST_TIMEOUT)
        if health_response.status_code != 200:
            logging.warning(f"健康检查失败，状态码: {health_response.status_code}")
            return False
        
        # 再测试生成端点
        generate_url = f"http://{VLLM_HOST}:{VLLM_PORT}/generate"
        test_payload = {
            "prompt": "测试",
            "user_input": "测试提示词续写",
            "user_id": 0,
            "max_tokens": 10,
            "temperature": 1.0,
            "top_p": 0.9,
            "type": "文生图",
            "use_lora": True
        }
        
        generate_response = requests.post(generate_url, json=test_payload, timeout=REQUEST_TIMEOUT)
        if generate_response.status_code != 200:
            logging.warning(f"生成请求失败，状态码: {generate_response.status_code}")
            return False
            
        logging.info("服务健康检查通过")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"请求异常: {e}")
        return False
    except Exception as e:
        logging.error(f"检查服务时发生错误: {e}")
        return False

def kill_process():
    """杀死运行在指定端口的进程"""
    try:
        # 查找使用指定端口的进程
        cmd = f"lsof -i:{VLLM_PORT} -t"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        
        if result.stdout:
            pid = result.stdout.strip()
            logging.info(f"找到运行在端口 {VLLM_PORT} 上的进程 (PID: {pid})，正在终止...")
            
            # 发送SIGTERM信号
            subprocess.run(f"kill -15 {pid}", shell=True)
            
            # 等待几秒钟
            time.sleep(5)
            
            # 检查进程是否仍在运行，如果是，则使用SIGKILL
            check_result = subprocess.run(f"ps -p {pid}", shell=True, capture_output=True)
            if check_result.returncode == 0:
                logging.warning(f"进程 {pid} 没有响应 SIGTERM，正在使用 SIGKILL...")
                subprocess.run(f"kill -9 {pid}", shell=True)
                
            logging.info(f"进程已终止")
            return True
        else:
            logging.warning(f"未找到运行在端口 {VLLM_PORT} 上的进程")
            return False
    except Exception as e:
        logging.error(f"终止进程时发生错误: {e}")
        return False

def restart_service():
    """重启VLLM服务"""
    try:
        logging.info("正在重启VLLM服务...")
        
        # 确保先杀死现有进程
        kill_process()
        
        # 等待端口释放
        time.sleep(5)
        
        # 执行重启脚本
        result = subprocess.run(f"bash {RESTART_SCRIPT}", shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("VLLM服务重启命令已执行")
            
            # 给服务一些启动时间
            logging.info("等待服务启动...")
            time.sleep(60)
            
            # 检查服务是否已启动
            if check_health():
                logging.info("服务已成功重启")
                return True
            else:
                logging.error("服务重启后健康检查失败")
                return False
        else:
            logging.error(f"重启脚本执行失败: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"重启服务时发生错误: {e}")
        return False

def main():
    """主函数"""
    logging.info("VLLM服务监控已启动")
    
    restart_attempts = 0
    last_restart_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # 检查服务健康状态
            if not check_health():
                logging.warning("VLLM服务健康检查失败")
                send_error_msg_to_feishu(f"vllm-提示词补全-VLLM服务健康检查失败")
                
                # 检查是否可以尝试重启
                if current_time - last_restart_time > RESTART_COOLDOWN:
                    if restart_attempts < MAX_RESTART_ATTEMPTS:
                        restart_attempts += 1
                        last_restart_time = current_time
                        
                        logging.warning(f"尝试重启服务 (尝试 {restart_attempts}/{MAX_RESTART_ATTEMPTS})")
                        if restart_service():
                            restart_attempts = 0
                            logging.info("服务已成功重启")
                            send_error_msg_to_feishu(f"vllm-提示词补全-VLLM服务已成功重启")
                        else:
                            logging.error("服务重启失败")
                            send_error_msg_to_feishu(f"vllm-提示词补全-VLLM服务重启失败")
                    else:
                        logging.critical(f"达到最大重启尝试次数 ({MAX_RESTART_ATTEMPTS})，不再尝试自动重启")
                        logging.critical("请手动检查服务状态")
                        send_error_msg_to_feishu(f"vllm-提示词补全-VLLM服务达到最大重启尝试次数 ({MAX_RESTART_ATTEMPTS})，不再尝试自动重启")
                        # 发送警报通知（可根据需要实现）
                else:
                    cooling_time = int(RESTART_COOLDOWN - (current_time - last_restart_time))
                    logging.info(f"冷却中，{cooling_time} 秒后将重新尝试重启")
                    send_error_msg_to_feishu(f"vllm-提示词补全-VLLM服务冷却中，{cooling_time} 秒后将重新尝试重启")
            else:
                # 服务正常，重置重启尝试计数
                restart_attempts = 0
                
            # 等待下一次检查
            logging.info(f"等待 {CHECK_INTERVAL} 秒后进行下一次检查...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logging.info("收到终止信号，监控服务已停止")
            break
        except Exception as e:
            logging.error(f"监控过程中发生错误: {e}")
            try:
                send_error_msg_to_feishu(f"vllm-提示词补全-监控过程中发生错误: {e}")
            except Exception as e:
                logging.error(f"发送消息到飞书失败: {e}")

            time.sleep(60)  # 发生错误时等待较短时间后重试

if __name__ == "__main__":
    main() 