device=0
batch_size=16

checkpoint_dir='log'
logname='exp1'

model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
flagfile="${checkpoint_dir}/${logname}/config.txt"
vis_path="${checkpoint_dir}/${logname}/visualization/"

CUDA_VISIBLE_DEVICES=$device python predict.py --flagfile $flagfile --local_rank -1 \
    --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
    --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
    --test_dataset_path /path/to/test/dataset # --vis_pred --visualize_bbox --visualize_match

