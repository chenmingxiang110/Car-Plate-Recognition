# Car-Plate-Recognition

<img src="logs/demo.png" width="400" height="400">

The entire algorithm includes a car plate detection algorithm (using image segmentation) and a car plate recognition algorithm (CTC loss). The accuracy of this algorithm is 98.92% based on the data given.

The models can be downloaded from pan.baidu.com:

```
link:https://pan.baidu.com/s/1e8AtCfJ01fiu-vgJLRZ_ZQ  code:s05b
```

The car plate training data is acquired from

```
https://github.com/detectRecog/CCPD
```

Please cite the paper if you are willing to use the dataset

```
@inproceedings{xu2018towards,
  title={Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={255--271},
  year={2018}
}
```

The downloaded data should be put under the "./CCPD2019" directory.
