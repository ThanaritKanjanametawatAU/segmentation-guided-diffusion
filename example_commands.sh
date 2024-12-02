# Main training script
python main.py \
    --mode train \
    --model_type DDIM \
    --img_size 512 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --num_epochs 1000 \
    --save_image_epochs 20 \
    --save_model_epochs 50

python main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 128 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB \
    --eval_batch_size 4 \
    --eval_sample_size 4 \
    --checkpoint 14

python main.py \
    --mode train \
    --model_type DDIM \
    --img_size 128 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB \
    --train_batch_size 10 \
    --eval_batch_size 1 \
    --num_epochs 10 \
    --save_image_epochs 2 \
    --save_model_epochs 2 \
    --resume_epoch 4







#Old ones
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
