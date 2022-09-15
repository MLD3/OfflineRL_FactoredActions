# Offline RL with Factored Action Spaces - Anonymous NeurIPS 2022 submission

## Synthetic domains
- Bandit 2D (Sec 3.3.2)
- Chain 2D (Sec 3.1, 3.2)

## Sepsis simulator (Sec 4.1)
- Simulator based on publicly available code at https://github.com/clinicalml/gumbel-max-scm/tree/sim-v2
- The preparation steps are in `data-prep`, which include the simulator source code as well as several notebooks for dataset generation. The output is saved to data (ground-truth MDP parameters, ground-truth optimal policy, and optimal value functions). 
- Use `commands.sh` to run `datagen` noteboooks to generate various datasets. Each dataset of size N=10000 may take up to 30 minutes to generate. 
- 5 behavior policies, 10 replications each

## MIMIC-sepsis (Sec 4.2)
- Cohort extraction based on publicly available code at https://github.com/microsoft/mimic_sepsis
- RNN AIS based on publicly available code at https://github.com/MLforHealth/rl_representations/
