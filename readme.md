<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
    <h3 align="center">Franka Inference</h3>

  <p align="center">
    Teaching frankas hand to reach the desired location and rotation, from scratch
    <br />
    <a href="https://git.tu-berlin.de/erik.fischer98/autonomous-agents/-/blob/main/readme.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://git.tu-berlin.de/erik.fischer98/autonomous-agents/-/blob/main/documentation/franka_move_demo.webm">View Demo</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

As a part of the Autonomous Agents module at Tu Berlin, this project was created. Its goal was to experiment with `NVIDIA Omniverse`, or more specifically its module, `ISAAC GYM`, to use its capabilities of high-realism and high-precision simulation to train RL-models, controlling the robots actions.

![Franka_moving](https://git.tu-berlin.de/erik.fischer98/autonomous-agents/-/blob/main/documentation/franka_move_demo.gif)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
# Getting Started

## Prerequisites

Requires expensive hardware to run ISAAC GYM. Find the required hardware [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

For ease of use, we recommend that you set an alias to ISAAC's own python installation as follows:

```sh
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```

- If configured in IDE, allows it to detect the classes used in ISAAC GYM, thus being able to resolve all references
- Allows you to manually execute files without using the `run_*.sh` files by adding the file to execute after the PYTHON_PATH alias


## Installation

1. Install `NVIDIA Omniverse`. [This guide](https://docs.omniverse.nvidia.com/prod_install-guide/prod_install-guide/workstation.html) describes the installation process precisely.
2. Launch it and download `ISAAC GYM` in the "Exchange" tab.
3. Clone the repo
   ```sh
   git clone https://git.tu-berlin.de/erik.fischer98/autonomous-agents.git
   ```
4. Install all requirements 
   ```sh
   PYTHON_PATH -m pip install -r requirements.txt
   ```




<!-- USAGE EXAMPLES -->
## Usage

1. For training execute the `run_training.sh` file
2. To see the results execute the `run_inference.sh` file 

Note: The reposetory comes with a pre-trained model. To start from scratch, delete the ppo_franka.zip in the franka_move folder.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Machine Learning
## Model
We used [Proxmial Policy Optimization](https://openai.com/blog/openai-baselines-ppo/). The implementation we use is from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).
## Simulation
Each simulation step, the Franka robot instantly assumes the predicted configuration.
- Upside: Extremely fast simulation, the effect of any action can be eveluated after one step.
- Downside: When Franka tries to assume target states in **franka_inference.py**, its trajectory may collide with the default ground plane, causing Franka to stop. May be improved with better pathfinding (current: Default ISAAG Gym, not affected by machine learning).
## Observations
The observations have a length of 7. They consist of the position of the target cube (xyz-coordinate) and its orientation in quaternions. 
## Reward
The reward is given by the linear combination of the negative distance to the target and the angle between the orientation of the roboters hand and the target cube. This way we ensture not only that the target is reached but also that the hand has a sensible orientation.
Also, configurations colliding with itself/invalid configurations are penalised with a big malus.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Other
## Challenges
- During the entirety of the project only one Pc had the hardware requirements which allowed running ISAAC Gym. Exhausting for main developer, slowed down development speed
- ISAAC GYM is a massive simulator, containing lots of interfaces with little documentation and even less small projects. Its meant for bigger teams or projects with development time exceeding five months.
## Missed Opportunities
- We implemented **franka_move_task.py** with the interface to simulate multiple Frankas at the same time, but failed to integrate it in **franka_train.py**.
- ISAAG Gym can generate generated massive amounts of data used for machine learning (images with high resolution, correctly placed bounding boxed, desired result images for semantig segmentation, ...). Would have been perfect for a model using cameras, lidar or other sensors. Sadly, we lacked the time.
<!-- LICENSE -->
## License

Distributed under the MIT License. See `license.txt` for more information.


<!-- CONTACT -->
## Contact

- Erik Fischer - erik.fischer98@win.tu-berlin.de
- Georgios Zountsas - zountsas@campus.tu-berlin.de
- Mehmet Yasin Cifci - cifci@campus.tu-berlin.de


<!-- USEFUL LINKS -->
## Useful links

* A great example of what ISAAC Gym can be used for is the [RL games libary](https://github.com/Denys88/rl_games). Sadly, we found it two weeks before finishing the development cyle and were unable to draw much inspiration from it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>