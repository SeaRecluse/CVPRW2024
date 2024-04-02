
echo "----- start test"

python test.py --data_path ./orig_data/ --model_path "./runs-convnextv2-b" --save_path "./test-submit.txt" --data_type p1,p2.1,p2.2

echo "----- Testing is over!"
