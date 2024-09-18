# Minimum-Variance Policy Evaluation for Policy Improvement (UAI 2023) (Ported to latest tensorflow) (Moved from rllib to ray-rllib)

**NOT TO BE DISTRIBUTED**

This repository contains the implementation of the MBPExPI algorithm, based on OpenAI baselines.

## What's new

We provide the implementation of a novel algorithm, under the folder OptimalMis.

## Full install (tested on Dynamite)

```
conda env create -f environment.yml
pip install requirements.txt --no-cache-dir
pip install -e .
```


## Usage

The scripts accept a large number of optional command-line arguments. Typical usage is:

```
python3 run.py --env [env name] --seed [random seed] --max_iters [iters number] --policy [linear] --policy_init [zeros] --variance_init [variance value] --shift_return [True | False] --constant_step_size [step size value] --inner [inner ites] --capacity [capacity] --max_offline_iters [offline iters] --penalization [True | False] --delta [delta value]
```

For example, to run a PO2PE experiment on the cartpole environment:

```
python3 baselines/optimalMis/run.py --env rllab.cartpole --seed 0 --max_iters 500 --policy 'linear' --policy_init 'zeros' --capacity 1 --inner 1  --variance_init -1 --constant_step_size 1 --max_offline_iters 10 --penalization True --delta 0.75
```

To compare the previous results with POIS:

```
python3 baselines/pois/run.py --env rllab.cartpole --seed 0 --max_iters 500 --policy 'linear' --policy_init 'zeros' --variance_init -1 --delta 0.4
```

To compare the previous results with TRPO:

```
python3 baselines/trpo_mpi/run.py --env rllab.cartpole --seed 0 --num_episodes 100 --max_iters 500 --policy 'linear' --policy_init 'zeros' --variance_init -1 --max_kl 0.01
```

The results are saved in csv and tensorboard formats under the ./logs directory.
