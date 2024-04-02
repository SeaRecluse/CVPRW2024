
echo "----- start train"

echo "----- start p1 train"
python main.py --model convnextv2_base.fcmae_ft_in22k_in1k_384  --batch_size 32 --train_data_ratio 0.8 --drop_path 0.1 --smoothing 0.8 --epochs 100 --warmup_epochs 20 --lr 1e-4 --data_path ./data/p1/ --output_dir ./runs-convnextv2-b/p1/ 

echo "----- start p2.1 train"
python main.py --model convnextv2_base.fcmae_ft_in22k_in1k_384  --batch_size 32 --train_data_ratio 0.6 --drop_path 0.2 --smoothing 0.8 --epochs 50 --warmup_epochs 10 --lr 1e-4 --data_path ./data/p2.1/ --output_dir ./runs-convnextv2-b/p2.1/ 

echo "----- start p2.2 train"
python main.py --model convnextv2_base.fcmae_ft_in22k_in1k_384 --batch_size 32 --train_data_ratio 0.4 --drop_path 0.2 --smoothing 0.8 --epochs 50 --warmup_epochs 20  --lr 4e-5 --data_path ./data/p2.2/ --output_dir ./runs-convnextv2-b/p2.2/ 

echo "----- Training is over!"
