# Leveraging Factored Action Spaces for Efficient Offline RL in Healthcare
This repository contains the source code for replicating all experiments in the NeurIPS 2022 paper, "Leveraging Factored Action Spaces for Efficient Offline Reinforcement Learning in Healthcare".

Repository content:

- `synthetic` contains code for the toy problems used in the theory sections. 
- `sepsisSim` contains code to replicate the experiments on the sepsis simulator.
- `RL_mimic_sepsis` contains code to replicate the real-data experiments on sepsis management using MIMIC-III. 

If you use this code in your research, please cite the following publication:
```
@inproceedings{tang2022leveraging,
    title={Leveraging Factored Action Spaces for Efficient Offline Reinforcement Learning in Healthcare},
    author={Tang, Shengpu and Makar, Maggie and Sjoding, Michael and Doshi-Velez, Finale and Wiens, Jenna},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=Jd70afzIvJ4}
}
```

## Synthetic domains
- Bandit 2D (Sec 3.3.2)
- Chain 2D (Sec 3.1, 3.2)

## Sepsis simulator (Sec 4.1)
- Simulator based on publicly available code at https://github.com/clinicalml/gumbel-max-scm/tree/sim-v2
- The preparation steps are in `data-prep`, which include the simulator source code as well as several notebooks for dataset generation. 
- Part of the output is saved to `data` (ground-truth MDP parameters, ground-truth optimal policy, and optimal value functions). 
- Use `commands.sh` to run `datagen` noteboooks to generate various datasets. Each dataset of size N=10000 may take up to 30 minutes to generate. 
- Each setting has 10 replications
- Fig 6 (from left to right) corresponds to the 5 behavior policies: 
    - eps_0_1: ε-greedy with ε=0.1
    - eps_0_5: ε-greedy with ε=0.5
    - eps_1: uniformly random policy
    - suboptimal99: take the optimal action w.p. ρ=0.01; randomly sample the remaining actions w.p. 0.99
    - suboptimal: never take the optimal action; randomly sample the remaining actions

## MIMIC-sepsis (Sec 4.2)
- Cohort extraction based on publicly available code at https://github.com/microsoft/mimic_sepsis
- RNN AIS based on publicly available code at https://github.com/MLforHealth/rl_representations/
