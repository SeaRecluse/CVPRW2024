
echo "----- start test"

python test.py --data_val_path ./orig_data/data/dev --data_test_path ./orig_data/HySpeFAS_test/images --model_path ./runs-convnextv2-SSI

echo "----- Testing is over!"
