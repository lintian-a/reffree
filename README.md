# RefFree: Reference-Free 3D Reconstruction of brain dissection photographs with ML
[![arXiv](https://img.shields.io/badge/arXiv-2403.05780-b31b1b.svg)](https://arxiv.org/abs/2503.09963)

**Reference-Free 3D Reconstruction of Brain Dissection Photographs with Machine Learning**  
Lin Tian, Sean I. Young, Jonathan Williams Ramirez, Dina Zemlyanker, Lucas Jacob Deden Binder, Rogeny Herisse, Theresa R. Connors, Derek H. Oakley, Bradley T. Hyman, Oula Puonti, Matthew S. Rosen, Juan Eugenio Iglesias [https://arxiv.org/abs/2403.05780](https://www.arxiv.org/abs/2503.09963)  

<div align="center">
<img src="figures/refree_teaser.png" width=550 alt="teaser">
</div>

RefFree is a deep learning-based ex vivo brain dissection photograph reconstruction pipeline. It consists of two stages: (1) an MNI coordinate prediction neural network trained with a dissection photograph synthetic data engine and (2) an MNI coordinate-based reconstruction algorithm, as shown below.


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

## 📄 Citation
If you find this repository useful, please consider cite:
```
@article{tian2025reference,
  title={Reference-Free 3D Reconstruction of Brain Dissection Photographs with Machine Learning},
  author={Tian, Lin and Young, Sean I and Ramirez, Jonathan Williams and Zemlyanker, Dina and Binder, Lucas Jacob Deden and Herisse, Rogeny and Connors, Theresa R and Oakley, Derek H and Hyman, Bradley T and Puonti, Oula and others},
  journal={arXiv preprint arXiv:2503.09963},
  year={2025}
}
```
