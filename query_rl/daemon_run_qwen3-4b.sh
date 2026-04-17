#!/bin/bash

# 定义你的原始启动脚本路径
INNER_SCRIPT="/home/yiao/verl/query_rl/run_qwen3-4b_search.sh"
# 定义日志路径
LOG_FILE="/home/yiao/verl/query_rl/log/run4.log"

echo "--- 守护进程已启动，时间: $(date) ---" >> $LOG_FILE

while true
do
    echo "--- 正在尝试启动/重启任务: $(date) ---" >> $LOG_FILE
    
    # 运行你的脚本 (注意这里不要加 nohup 或 &，因为我们需要脚本阻塞并监控它)
    sh $INNER_SCRIPT >> $LOG_FILE 2>&1
    
    # 获取上一个命令的退出状态码
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "--- 程序正常结束 (Exit Code 0)，停止重启。时间: $(date) ---" >> $LOG_FILE
        break
    else
        echo "--- 检测到异常退出 (Exit Code $EXIT_CODE)，可能是 OOM。10秒后自动重启... ---" >> $LOG_FILE
        # 强制释放一下当前进程可能残留在显存中的碎片（可选）
        sleep 10
    fi
done