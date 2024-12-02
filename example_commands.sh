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

python main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB \
    --train_batch_size 10 \
    --eval_batch_size 1 \
    --num_epochs 1000 \
    --save_image_epochs 5 \
    --save_model_epochs 50

# Sampling
python main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset breast_mri \
    --eval_batch_size 8 \
    --eval_sample_size 100