## About This Project
This project is part of our work for course ROBO.666 Robotics Project Work. This project enables developers and researchers to train Safety Gymnasium models with omnisafe algorithms. The project has two main python modules: train_with_agent.py and play_policy.py. The first is used to train a model, while the second is used to render a video of the model in action.

## Set Up
We recommend using a linux machine with Ubuntu-22.04 distribution. For a windows machine, dual boot with a linux distribution or install windows subsystem for linux [(WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). With dual boot the training times are significantly faster. It takes around 4 hours on WSL to train the default 2 million steps, but on dual boot this only takes around 30 minutes.

When using VS Code, this project can be utilized and developed using a devcontainer, which is a docker container with a predefined image. This makes setup much easier, since all the dependencies will be installed automatically.

To use the dev container setup, open VS Code and install [Dev Containers extension.](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Then open VS Code again while at the root of this repo. You'll be prompted with a message that asks if you'd like to open the project in a container. Click yes. VS Code will then build the docker image. This will take a moment. Once all dependencies are installed, all the files are loaded and you can open a new terminal to run the python modules inside the container.

If you wish to not use the devcontainers, follow the these steps.

1. Make sure you are using Ubuntu-22.04 distribution.

2. Get apt packages:

```
apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    xvfb \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libosmesa6-dev \
    python3 \
    python3-pip \
    python3-venv
```

3. Setup pip
```
python3 -m pip install --upgrade pip wheel setuptools
```

4. Install PIP Packages
```
pip install \
    numpy==1.23.0 \
    mujoco==2.3.3 \
    gymnasium \
    safety-gymnasium \
    omnisafe \
    wandb weave
```

5. Install other useful Python packages
```
pip install \
    jupyter \
    matplotlib \
    seaborn \
    tensorboard \
    ipykernel
```

## Setting Up W&B
W&B is a platform that helps developing AI models. For this project we used it to track the models in real-time during training. This section details the steps to link the project to the W&B platform. However, this is optional. The scripts in this project can be used without W&B.

1. Go to [W&B's website](https://wandb.ai/site/) and create and account.
2. From the sidebar, create a new project.
3. You'll be redirected to a quickstart guide. Follow the instructions to create an API token. Save the token.
4. Login to W&B from terminal:
```
wandb login
```

## Notes
If you have any questions, you can message the project contributors.

This project builds on top two ground breaking research papers and we give credit to their authors:

[Safety Gymnaisum](https://proceedings.neurips.cc/paper_files/paper/2023/file/3c557a3d6a48cc99444f85e924c66753-Paper-Datasets_and_Benchmarks.pdf)
Ji, Jiaming, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Juntao Dai, and Yaodong Yang. 2023. Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark. In Advances in Neural Information Processing Systems 37 (NeurIPS 2023) — Datasets and Benchmarks Track.

[Omnisafe](https://arxiv.org/pdf/2305.09304)
Ji, Jiaming, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, and Yaodong Yang. 2023. OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research. arXiv preprint arXiv:2305.09304. https://arxiv.org/abs/2305.09304