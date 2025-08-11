#!/bin/bash

cd /root/malio/vllm
# cd /home/zhangxuqi/malio/code/vllm

/root/miniconda3/envs/vllm/bin/python check_vllm_service.py > check.log 2>&1 & 
