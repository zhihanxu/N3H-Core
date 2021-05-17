export CUDA_VISIBLE_DEVICES=3
python rl_quantize.py               \
 --arch res18_image              \
 --dataset imagenet100              \
 --dataset_root data/imagenet100    \
 --suffix res18_image100_06              \
 --preserve_ratio 0.6              \
 --float_bit 8                      \
 --max_bit 8                        \
 --min_bit 2                        \
 --n_worker 32                      \
 --train_episode 1200                \
 --data_bsize 256                   \
 --linear_quantization              \
 --customize                        \
