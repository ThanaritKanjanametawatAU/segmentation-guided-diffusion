# Main training script
python main.py \
    --mode train \
    --model_type DDIM \
    --img_size 128 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB-v2 \
    --train_batch_size 16 \
    --eval_batch_size 1 \
    --num_epochs 2000 \
    --save_image_epochs 100 \
    --save_model_epochs 200 

# Evaluation
python main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 128 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB-v2 \
    --eval_batch_size 4 \
    --eval_sample_size 4 \
    --checkpoint 5000

# Resume Training
python main.py \
    --mode train \
    --model_type DDIM \
    --img_size 128 \
    --num_img_channels 3 \
    --dataset Thanarit/PCB \
    --train_batch_size 10 \
    --eval_batch_size 1 \
    --num_epochs 15 \
    --save_image_epochs 2 \
    --save_model_epochs 2 \
    --resume_epoch 14









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
