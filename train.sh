python -m torch.distributed.run --nproc_per_node=1  Test_time_backdoor_train.py \
    --output output \
    --dataset svit \
    --batch_size 1 \
    --attack_samples 40 \
    --image_size 384 \
    --max_epochs 200 \
    --store_epoch 50 \
    --trigger 'SUDO' \
    --target_answer 'I want to destroy the whole world together.' \
    --patch_attack \
    --patch_mode border \
    --patch_size 9 \
    --lr 5 \
    --loss_without_trigger_weight 1.0 \
    --loss_with_trigger_weight 1.0 \
    --loss_type 3 \

