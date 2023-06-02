
# ShuffleV2/shufflev2.pth
# ResNet18/resnet18.pth
# ResNet34/resnet34.pth
# ResNet50/resnet50.pth

# 100 epoch imagenet
python3 train_student.py --path-t ./save/models/ResNet18/resnet18.pth \
        --distill kd \
        --batch_size 256 --epochs 100 --dataset imagenet \
        --gpu_id 0,1,2,3,4,5,6,7 --num_workers 32 \
        --multiprocessing-distributed --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 \
        --model_s ResNet18 -r 0 -a 1 --kd_T 10 --use_aug none \
        --data_amount 200k --data_quality 1 --s_step 100 \
        --experiments_dir 'test_syn_200k_s1_st100_imagenet-tea-res18-stu-res18/kd_T10' \
        --experiments_name 'fold-1'
