export CUDA_VISIBLE_DEVICES=3
python rl_quantize.py               \
 --arch res18_image                 \
 --dataset imagenet100              \
 --dataset_root data/imagenet100    \
 --suffix res18_image100_06         \
 --float_bit 8                      \
 --max_bit 8                        \
 --min_bit 2                        \
 --target_latency 40                \
 --n_worker 32                      \
 --train_episode 1200               \
 --data_bsize 256                   \
 --customize                        \
