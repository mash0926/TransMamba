export CUDA_VISIBLE_DEVICES=0

python -u run_TransMamba.py \
--d_model 64 \
--dataset PSM \
--channel 25 \
--win_size 90 \
--patch_size_high 1 3 \
--patch_size_low 2 6 \
--mode train \

python -u run_TransMamba.py \
--anomaly_ratio 1 \
--d_model 64 \
--dataset PSM \
--channel 25 \
--win_size 90 \
--patch_size_high 1 3 \
--patch_size_low 2 6 \
--mode test