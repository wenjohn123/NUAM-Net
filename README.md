# NUAM-Net: A Novel Underwater Image Enhancement Attention Mechanism Network (JMSE 2024)([Paper](https://arxiv.org/pdf/2207.09689.pdf))
The Pytorch Implementation of ''NUAM-Net: A Novel Underwater Image Enhancement Attention Mechanism Network''. 

<div align=center><img src="img/1.png" height = "60%" width = "60%"/></div>

## Introduction
In this project, we use Python 3.7, Pytorch 1.7.1 and one NVIDIA RTX 3090 GPU. 

## Running

### Testing

Download the pretrained model [pretrained model].

Check the model and image pathes in Test_MC.py and Test_MP.py, and then run:

```
python Test_MC.py
```
```
python Test_MP.py
```

### Training

To train the model, you need to prepare our [dataset](https://drive.google.com/file/d/1YXdyNT9ac6CCpQTNKP7SnKtlRyugauvh/view?usp=sharing).

Check the dataset path in Train.py, and then run:
```
python Train.py
```

## Citation

If you find NUAM-Net is useful in your research, please cite our paper:

```
@article{wen2024nuam,
  title={NUAM-Net: A Novel Underwater Image Enhancement Attention Mechanism Network},
  author={Wen, Zhang and Zhao, Yikang and Gao, Feng and Su, Hao and Rao, Yuan and Dong, Junyu},
  journal={Journal of Marine Science and Engineering},
  volume={12},
  number={7},
  pages={1216},
  year={2024},
  publisher={MDPI}
}
```

