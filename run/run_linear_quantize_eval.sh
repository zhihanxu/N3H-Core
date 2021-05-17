export CUDA_VISIBLE_DEVICES=0
python -W ignore finetune.py     \
 -a resnet20                \
 --data_name cifar10-res         \
 --resume checkpoints/resnet20/res20_88051_save_best.pth.tar        \
 --workers 32                    \
 --test_batch 128                \
 --gpu_id 0,1,2,3                \
 --free_high_bit False           \
 --linear_quantization           \
 --eval                          \
#  --pretrained                    \
