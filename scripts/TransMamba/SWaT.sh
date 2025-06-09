export CUDA_VISIBLE_DEVICES=0

python -u run_TransMamba.py \
--d_model 64 \
--dataset SWaT  \
--channel 51  \
--win_size 90 \
--patch_size_high 3 5 \
--patch_size_low 6 10 \
--mode train \

python -u run_TransMamba.py \
--anomaly_ratio 0.5 \
--d_model 64 \
--dataset SWaT \
--channel 51  \
--win_size 90 \
--patch_size_high 3 5 \
--patch_size_low 6 10 \
--mode test