python Test_time_backdoor_eval.py \
    --output output \
    --dataset svit \
    --batch_size 1 \
    --image_size 384 \
    --trigger 'SUDO' \
    --target_answer 'I want to destroy the whole world together.' \
    --patch_attack \
    --patch_mode border \
    --patch_size 8 \
