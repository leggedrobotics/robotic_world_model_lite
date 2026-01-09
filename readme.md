# Robotic World Model Lite

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


This repository is a lightweight implementation intended for users who want to train dynamics models from offline data and model-based policies only, without the need to set up a full robotics simulator like [Isaac Lab](https://isaac-sim.github.io/IsaacLab).

**We provide a Google Colab notebook for quick start: [RWM Lite Colab Notebook](https://colab.research.google.com/drive/1SRL0ss59RxMp-MwY38Pi6iRW-VTFeku6?usp=sharing).**

:star:
For the full version with online simulator-based data collection, model and policy training and evaluation pipeline, please refer to our full [Isaac Lab RWM Extension](https://github.com/leggedrobotics/robotic_world_model) implementation.


## Overview

This repository provides a lightweight training pipeline for
- [**Robotic World Model (RWM)**](https://sites.google.com/view/roboticworldmodel/home),
- [**Uncertainty-Aware Robotic World Model (RWM-U)**](https://sites.google.com/view/uncertainty-aware-rwm),

and related model-based reinforcement learning methods, *without any simulator dependency*.

It enables:
- training of dynamics models with ensemble recurrent neural networks,
- training of policies with learned neural network dynamics without any simulator,
- WandB logging support for experiment tracking.

<table>
  <tr>
  <td valign="top" width="50%">

![Robotic World Model](rwm.png)

**Paper**: [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)  
**Project Page**: [https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel)

  </td>
  <td valign="top" width="50%">

![Uncertainty-Aware Robotic World Model](rwm-u.png)

**Paper**: [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)  
**Project Page**: [https://sites.google.com/view/uncertainty-aware-rwm](https://sites.google.com/view/uncertainty-aware-rwm)

  </td>
  </tr>
</table>

**Authors**: [Chenhao Li](https://breadli428.github.io/), [Andreas Krause](https://las.inf.ethz.ch/krausea), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)  
**Affiliation**: [ETH AI Center](https://ai.ethz.ch/), [Learning & Adaptive Systems Group](https://las.inf.ethz.ch/) and [Robotic Systems Lab](https://rsl.ethz.ch/), [ETH Zurich](https://ethz.ch/en.html)


---


## Installation

1. **Create Conda environment** with `python>=3.10` and activate it

```bash
conda create -n rwm_lite python=3.10 -y
conda activate rwm_lite
```

2. **Clone this repository** inside your Isaac Lab directory

```bash
git clone git@github.com:leggedrobotics/robotic_world_model_lite.git
cd robotic_world_model_lite
```

3. **Install `rwm_lite`**

```bash
python -m pip install -e .
```

## Model-Based Policy Training & Evaluation

1. **Login WandB**

```bash
wandb login
```

2. **Train policy with RWM**

```bash
python scripts/train.py --task anymal_d_flat
```

The policy is saved under `logs/`.

3. **Evaluate the policy with a simulator or hardware**

The learned policy can be played and evaluated with our full [Isaac Lab RWM Extension](https://github.com/leggedrobotics/robotic_world_model) or the original [Isaac Lab](https://isaac-sim.github.io/IsaacLab) task registry.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-D-Play-v0 --checkpoint <checkpoint_path>
```

---

## Code Structure

We provide a reference pipeline that enables RWM and RWM-U on ANYmal D.

Key files:

- Environment configurations + Imagination rollout logic (constructs policy observations & rewards from model outputs)
  [`anymal_d_flat.py`](scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py).
- Algorithm configuration + training parameters
  [`anymal_d_flat_cfg.py`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).
- Pretrained RWM-U checkpoint
  [`pretrain_rnn_ens.pt`](assets/models/pretrain_rnn_ens.pt).
- Initial states for imagination rollout
  [`state_action_data_0.csv`](assets/data/state_action_data_0.csv).


---

## Citation
If you find this repository useful for your research, please consider citing:

```text
@article{li2025robotic,
  title={Robotic world model: A neural network simulator for robust policy optimization in robotics},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2501.10100},
  year={2025}
}
@article{li2025offline,
  title={Offline Robotic World Model: Learning Robotic Policies without a Physics Simulator},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2504.16680},
  year={2025}
}
```
