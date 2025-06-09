export CUDA_VISIBLE_DEVICES=0

python -u run_TransMamba.py \
--d_model 64 \
--dataset SMAP  \
--channel 25    \
--win_size 240 \
--patch_size_high 3 5 \
--patch_size_low 6 10 \
--mode train \

python -u run_TransMamba.py \
--anomaly_ratio 0.725 \
--d_model 64 \
--dataset SMAP  \
--channel 25    \
--win_size 240 \
--patch_size_high 3 5 \
--patch_size_low 6 10 \
--mode test