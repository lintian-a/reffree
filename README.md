# RefFree: Reference-Free 3D Reconstruction of brain dissection photographs with ML
Official Implementation for "Reference-Free 3D Reconstruction of  Brain Dissection Photographs with Machine Learning"

<div align="center">
<img src="figures/refree_teaser.png" width=550 alt="teaser">
</div>

RefFree is a deep learning-based ex vivo brain dissection photograph reconstruction pipeline. It consists of two stages: (1) a MNI coordinate prediction neural network trained with a dissection photograph synthetic data engine, and (2) an MNI coordinate-based reconstruction algorithm as shown below.


<div align="center">
<img src="figures/refree_pipeline.png" width=400 alt="teaser">
</div>

TODO:
- [ ] Release data processing code

## 👉 Installation
Run the following 
```
conda create -n reffree python==3.9
conda activate reffree

cd reffree
pip install .
```

## 👉 Data preprocessing

## 👉 Training
Before training, create a train_config.py file from configs/train_config_example.py. Update the directory setting in train_config.py.

Train the model with the following command line
```
python src/reffree/train.py -e [EXP_ID] --exp_folder [EXP_FOLDER] --batch_size 2 --slice_num 32 --epoch 100 --train_config ./configs/train_config.py
```


