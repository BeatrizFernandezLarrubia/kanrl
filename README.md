# SCALABILITY AND ROBUSTNESS OF KANs APPLIED TO REINFORCEMENT LEARNING

## Deep Leaning Lab at Universität Freiburg (Summer Semester 2024)

### Members: Beatriz Fernández Larrubia, Ayisha Ryhana Dawood, Pojen Shih
### Supervisor: Baohe Zhang

KANs [2] are integrated into the reinforcement learning framework by replacing the MLP component of Deep Q-Networks (DQN), resulting in the Kolmogorov-Arnold Q-Network (KAQN) [3].
The three points of interest were:
1.Comparing the performance of KANs and MLPs in environments with varying numbers of distractors.
2.Evaluating how KANs compare against MLPs in environments with high complexity. (Use MorphingAnt [8] to add arbitrary number of legs to the Ant.)

The code was initially forked from [3]. To mitigate the lengthy processing durations of PyKAN [2], EfficientKAN [5] was adopted to ensure more time-efficient execution.

### Run Ant with multiple legs
Modify config.yaml with the necessary parameters and run the file run.py - `python -m run`.

The run.py file has the following functionalities:
1. Train and/or test Ant with arbitrary number of legs.
2. Render the KAN graph with the target_network.pt file of an already trained agent.
3. Test multiple actor.pt files, calculate and write mean and standard deviation of the rewards to .csv files.

The plot_rewards.py file can be used to plot the results of the training or testing of the agents.

## References

[1] Liu, Wang, et al. "Kan: Kolmogorov-arnold networks." arXiv:2404.19756 (2024). 
[2] KAN Github: https://github.com/KindXiaoming/pykan 
[3] KANrl github: https://github.com/riiswa/kanrl 
[4] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015). 
[5] EfficientKAN github: https://github.com/Blealtan/efficient-kan 
[6] Ant-v4: https://gymnasium.farama.org/environments/mujoco/ant/ 
[7] Ni, Tianwei, et al. "Bridging State and History Representations: Understanding Self-Predictive RL." arXiv preprint arXiv:2401.08898 (2024). 
[8] MorphingAnt github: https://github.com/brandontrabucco/morphing- agents/blob/master/morphing_agents/mujoco/ant/env.py


# Original README.md from kanrl [3]
# Kolmogorov-Arnold Q-Network (KAQN) - KAN applied to Reinforcement learning, initial experiments

This small project test the novel architecture Kolmogorov-Arnold Networks (KAN) in the reinforcement learning paradigm to the CartPole problem. 

KANs are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model accuracy and interpretability.

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

For more information about this novel architecture please visit:
- The official Pytorch implementation of the architecture: https://github.com/KindXiaoming/pykan
- The research paper: https://arxiv.org/abs/2404.19756

## Experimentation

The implementation of Kolmogorov-Arnold Q-Network (KAQN) offers a promising avenue in reinforcement learning. In this project, we replace the Multi-Layer Perceptron (MLP) component of Deep Q-Networks (DQN) with the Kolmogorov-Arnold Network. Furthermore, we employ the Double Deep Q-Network (DDQN) update rule to enhance stability and learning efficiency.

The following plot compare DDQN implementation with KAN (width=8) and the classical MLP (width=32) on the `CartPole-v1` environment for 500 episodes on 32 seeds (with 50 warm-ups episodes).

<img alt="Epsisode length evolution during training on CartPole-v1" src="https://raw.githubusercontent.com/riiswa/kanrl/main/cartpole_results.png">

The following plot displays the interpretable policy learned by KAQN during a successful training session.

<img alt="Interpretable policy for CartPole" src="https://raw.githubusercontent.com/riiswa/kanrl/main/policy.png">

- **Observation**: KAQN exhibits unstable learning and struggles to solve `CartPole-v1` across multiple seeds with the current hyperparameters (refer to `config.yaml`).
- **Next Steps**: Further investigation is warranted to select more suitable hyperparameters. It's possible that KAQN encounters challenges with the non-stationary nature of value function approximation. Consider exploring alternative configurations or adapting KAQN for policy learning.
- **Performance Comparison**: It's noteworthy that KAQN operates notably slower than DQN, with over a 10x difference in speed, despite having fewer parameters. This applies to both inference and training phases.
- **Interpretable Policy**: The learned policy with KANs is more interpretable than MLP, I'm currently working on extraction on interpretable policy...


### KAN for RL interpretability

In a web application, we showcase a method to make a trained RL policy interpretable using KAN. The process involves transferring the knowledge from a pre-trained RL policy to a KAN. The process involve:
- Train the KAN using observations from trajectories generated by a pre-trained RL policy, the KAN learns to map observations to corresponding actions.
- Apply symbolic regression algorithms to the KAN's learned mapping.
- Extract an interpretable policy expressed in symbolic form.

To launch the app (after cloning the repo), run : 

```bash
cd interpretable
python app.py
```

There is also a live-demo on Hugging Face : https://huggingface.co/spaces/riiswa/RL-Interpretable-Policy-via-Kolmogorov-Arnold-Network

<img alt="demo" src="https://raw.githubusercontent.com/riiswa/kanrl/main/interpretable/demo.gif">

## Contributing

I welcome the community to enhance this project. There are plenty of opportunities to contribute, like hyperparameters search, benchmark with classic DQN, implementation of others algorithm (REINFORCE, A2C, etc...) and additional environment support.
Feel free to submit pull requests with your contributions or open issues to discuss ideas and improvements. Together, we can explore the full potential of KAN and advance the field of reinforcement learning ❤️.

