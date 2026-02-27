nohup python /home/yiao/verl/query_rl/retrieval_server.py > ~/verl/query_rl/log/retrieval_server.log 2>&1 &
nohup sh /home/yiao/verl/query_rl/embeding_server.sh > /home/yiao/verl/query_rl/log/embeding_server.log 2>&1 &
nohup sh /home/yiao/verl/query_rl/run_qwen3-0.6b_search_multiturn2.sh > /home/yiao/verl/query_rl/log/run2.log 2>&1 &
