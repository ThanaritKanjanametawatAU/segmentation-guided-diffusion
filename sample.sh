# Training
python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 640 \
    --num_img_channels 3 \
    --dataset printed_circuit_board \
    --img_dir Data \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --num_epochs 1000

# Sampling
python main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset breast_mri \
    --eval_batch_size 8 \
    --eval_sample_size 100