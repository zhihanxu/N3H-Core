export CUDA_VISIBLE_DEVICES=1
python -W ignore finetune.py     \
 -a res18_image               \
 -c checkpoints/res18_image     \
 --data_name ILSVRC2012      \
 --data data/imagenet/imagenet          \
 --epochs 90                     \
 --lr 0.0005                     \
 --train_batch 256               \
 --wd 4e-5                       \
 --workers 2                    \
 --linear_quantization           \
 --customize                     \
 --pretrained                    \


