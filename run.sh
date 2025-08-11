cd /root/malio/vllm

nohup  /root/miniconda3/envs/vllm/bin/python  92-服务器部署vllm服务.py --port 28000 --gpu-memory-utilization 0.3 >> run.log 2>&1 &
