import requests
import logging

logger = logging.getLogger(__name__)

def send_error_msg_to_feishu(msg):
    # 定义请求的URL和数据
    url = "https://open.feishu.cn/open-apis/bot/v2/hook/e46131fc-8628-41fe-928a-2d5605e8da11"
    data = {
        "msg_type": "text",
        "content": {
            "text": msg
        }
    }
    # 发送POST请求
    try:
        response = requests.post(url, json=data, timeout=1)

        if response.status_code != 200:
            logger.error(f"发送消息到飞书失败，状态码：{response.status_code}")
        else:
            logger.info(f"发送消息到飞书成功, error_msg: {msg}")

    except requests.exceptions.RequestException as e:
        logger.error(f"发送消息到飞书失败，错误信息: {e}")

if __name__ == "__main__":
    send_error_msg_to_feishu("测试消息")