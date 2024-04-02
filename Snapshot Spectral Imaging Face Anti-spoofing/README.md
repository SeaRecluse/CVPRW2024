# 5th Chalearn Face Anti-spoofing Workshop and Challenge@CVPR2024 —— Team SeaRecluse

## Step
### Install dependencies:
```bash
pip install -r requirements.txt
```

### Data preprocessing:
```
If you have full content data data, it should look like this in the folder
--orig_data
    -HySpeFAS_trainval*
        -images *
        -train.txt *
        -val_gt.txt *
    -HySpeFAS_test*
        -images *
       
    -data_sorting.py
    -data_processing.py
    -data_handle.sh

```
Execute data_handle.sh for data sorting and preprocessing
``` shell
cd orig_data
sh ./data_handle.sh
```

### Start training
```
# We have usd a pre-trained model based on ImageNet provided by timm. Please ensure that your network is accessible to download this model.
# Put the enhanced data into the folder "./data" for training.
# The default batch size is 32, which requires at least 24G GPU memory for training.
```
train for all
```
sh ./train.sh
```

### Test your model
```
# When the training is over, you can execute test.py and use it for testing.

sh ./test.sh
```

### Visualize your predicted score domain
```
# When the training is over, you can execute show_datazone.py and use it for draw data domain about your predicted score.

python show_data_zone.py
```

## other
The Flops requirement of this competition model is less than 100G, but this is still a huge number. If the organizer has sufficient training resources and can use a larger backbone, we believe that better results will be achieved.

