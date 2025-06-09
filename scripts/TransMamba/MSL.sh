export CUDA_VISIBLE_DEVICES=0

python -u run_TransMamba.py \
--d_model 64 \
--dataset MSL  \
--channel 55   \
--win_size 90 \
--patch_size_high 2 3 5 \
--patch_size_low 6 9 15 \
--mode train \

python -u run_TransMamba.py \
--anomaly_ratio 1 \
--d_model 64 \
--dataset MSL \
--channel 55 \
--win_size 90 \
--patch_size_high 2 3 5 \
--patch_size_low 6 9 15 \
--mode test