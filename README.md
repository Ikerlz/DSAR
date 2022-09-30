# DSAR

## Introduction

- This is the code for the paper "xxxxx"

- There are two parts in the repository, one is for the simulation, which is the matlab codes and the other one is for the real case.

- You can get all datasets and environments from "xxx"

## Structure

├── Real Case
│   ├── code
│   │   ├── dsar.py
│   │   ├── global_algorithm.py
│   │   └── run_spark_yarn.sh
│   └── logs
│       ├── estimate_onestep_map.csv
│       ├── estimate_twostep_map.csv
│       ├── inference_onestep_map.csv
│       └── inference_twostep_map.csv
└── Simulation
    └── matlab_code
        ├── cal_Xi.m
        ├── dsar.m
        ├── estimate.m
        ├── generate_dist_data.m
        ├── generate_dist_data_main.m
        ├── mainblock.m
        ├── main.m
        ├── matrixAblock.m
        ├── partfour.m
        ├── partone.m
        ├── partthree.m
        ├── parttwo.m
        ├── partzero.m
        ├── PowerLaw.m
        ├── SBM.m
        ├── T1.m
        ├── T2.m
        ├── T3.m
        ├── test.m
        ├── U1.m
        ├── U2.m
        ├── V1.m
        ├── V2.m
        └── Xi0.m
