# Stable-Time Path Planning

## 1. Installation

To reproduce our ablation results, please install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html#) environment on a Linux machine with Nvidia GPU.<br>
You may need to install the following apt packages:
```bash
sudo apt-get install libboost-dev
```
Please config the conda environment:
```bash
cd ~/DPtraj/deepPathPlan
conda create -n <your_env_name> python=3.8
conda activate <your_env_name>
pip install -r requirements.txt
```


## 2. Reproducing the Model
### 2.1 Download Training Data
Please download, unzip and place the [Data](https://drive.google.com/file/d/1uuQsWTBYMzHI0RcXgpFg6Ft-3lf6-fRD/view?usp=drive_link) as the directory `~/totalData`.

### 2.2 Training
You can easily retrain the model by running `PathNet/train_ours.py`:

```bash
cd ~/DPtraj/deepPathPlan
python PathNet/train_ours.py
```

### 2.3 Visualizing the Results
Once the model converges, you can visualize it. A pretrained model is provided [here](https://drive.google.com/file/d/13o9flu4yo451FzMRhiEFF8PYq2dcNoDo/view?usp=sharing):

```bash
cd ~/DPtraj/deepPathPlan
python PathNet/visualizer_tojit.py
```

> **Note:** This script will utilize `torch.jit.trace` to generate a model file that can be directly invoked by LibTorch, allowing you to seamlessly integrate it into our ROS program.

## 3. Checkout our experiment logs
To check similar results in table Table.S1 and Fig.S2 of Supplementary Materials, we provide:<br>
1. Detailed eval log [`model.pkl.txt`](deepPathPlan/models/model.pkl.txt).
2. Detailed training log [`model.pklStep.txt`](deepPathPlan/models/model.pklStep.txt).
3. Reproduced model [`model.pkl`](deepPathPlan/models/model.pkl).

> **Note:**  
> Please note that slight differences in the results compared to the paper are normal, due to variations in training configuration and package versions.<br>
> For example, the batch size (`bz`) is set to 32 in this repo for easier reproduction on GPUs with 16 GB memory (the bz used in the paper is 64 based on 32 GB memory).  


## 4. Contact

If you have any questions, please feel free to contact Zhichao HAN (<zhichaohan@zju.edu.cn>) or Mengze TIAN(<mengze.tian@epfl.ch>).
