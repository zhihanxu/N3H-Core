# N3H-Core
Code for N3H-Core adapted from HAQ (CVPR 19')

## Dependencies
Current code base is tested under following environment:  
1. Python 3.8.5
2. Pytorch 1.6.0
3. torchvision 0.7.0
4. numpy 1.19.2
5. progress 1.5
6. tensorboardX 2.1

## Dataset
Create a link of the imagenet in the current folder:  
`ln -s /path/to/imagenet/ data/`

Prepare imagenet100 dataset for fast search process:  
`python lib/utils/make_data.py`

## RL-based Joint Search
Set target latency in the bash file:
`--target_latency 35`

Run the bash file:  
`bash run/run_linear_quantize_search.sh`
or (save the log file):
`nohup bash run/run_linear_quantize_search.sh > search_tar35.log 2>&1 &`

## Retrain and Finetune
Copy the best policy (quantization + workload split-ratio) within the finetune.py:
`strategy = `