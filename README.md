# Multitasking Models for Molecular Property Prediction

This repository contains multitasking models for predicting molecular properties from the **QM9 dataset** ([C. Ramakrishnan et al., 2014](https://arxiv.org/abs/1409.1556)).  
The models are constructed using the [e3nn](https://e3nn.org/) package, which enables E(3)-equivariant representations.

## Models

Three distinct models are implemented:

1. **Singletasking**  
   Baseline model that predicts a single property.

2. **Multitasking**  
   Decision-tree-like model that predicts all 12 properties of the QM9 dataset.

3. **Charges**  
   A simplified multitasking model that predicts only the electric dipole moment and the spatial extent.

## Multi-task Loss Strategies

Two distinct strategies are implemented for managing the multi-task loss:

1. **Static Loss Weighting**  
   A standard approach using fixed weights for each task's loss.

2. **Dynamic Balance MTDL (DB-MTL)**  
   A sophisticated method from [B. Lin et al., 2023](https://arxiv.org/abs/2301.08128) for dynamically balancing losses across tasks.  
   This is a modified version of the implementation from [LibMTL](https://github.com/median-research-group/LibMTL).
