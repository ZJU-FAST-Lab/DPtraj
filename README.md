# DPtraj
A double-polynomial discription for trajectory interfaced with learning-based front end.

This work is presented in the paper: Hierarchically Depicting Vehicle Trajectory with Stability in Complex Environments, published in Science Robotics.

The backend trajectory optimizer improvements build upon our previous work (available at https://github.com/ZJU-FAST-Lab/Dftpav), where singularity issues were addressed. 

Moreover, the approach has recently been extended and applied to more complex multi-joint robotic platforms (see https://github.com/Tracailer/Tracailer).


If you find this repository helpful, please consider citing at least one of the following papers:

```bibtex
@article{han2025hierarchically,
  title={Hierarchically depicting vehicle trajectory with stability in complex environments},
  author={Han, Zhichao and Tian, Mengze and Gongye, Zaitian and Xue, Donglai and Xing, Jiaxi and Wang, Qianhao and Gao, Yuman and Wang, Jingping and Xu, Chao and Gao, Fei},
  journal={Science Robotics},
  volume={10},
  number={103},
  pages={eads4551},
  year={2025},
  publisher={American Association for the Advancement of Science}
}
@article{han2023efficient,
  title={An efficient spatial-temporal trajectory planner for autonomous vehicles in unstructured environments},
  author={Han, Zhichao and Wu, Yuwei and Li, Tong and Zhang, Lu and Pei, Liuao and Xu, Long and Li, Chengyang and Ma, Changjia and Xu, Chao and Shen, Shaojie and others},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={2},
  pages={1797--1814},
  year={2023},
  publisher={IEEE}
}
```

The code will be divided into several modules and gradually open-sourced in different branches. Currently, you can switch to the `backend` branch to try our efficient singularity-free backend optimization. This branch includes a README to guide you through quick deployment.