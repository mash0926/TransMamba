export CUDA_VISIBLE_DEVICES=0

python -u run_TransMamba.py \
--d_model 64 \
--dataset SMD  \
--channel 38    \
--win_size 240 \
--patch_size_high 3 5 \
--patch_size_low 5 15 \
--mode train \

python -u run_TransMamba.py \
--anomaly_ratio 0.5 \
--d_model 64 \
--dataset SMD  \
--channel 38    \
--win_size 240 \
--patch_size_high 3 5 \
--patch_size_low 5 15 \
--mode test