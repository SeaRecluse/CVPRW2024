
echo "----- start train"

echo "----- start SSI train"
python main.py --batch_size 64 --train_data_ratio 1.0 --drop_path 0.2 --smoothing 0.5 --epochs 200 --warmup_epochs 20  --lr 4e-5 --data_path ./data_extend --output_dir ./runs-convnextv2-SSI/extend-resize-gauss/ 

echo "----- Training is over!"
