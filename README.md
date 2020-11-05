# Efficient Marginalization of Discrete and Structured Latent Variables via Sparsity - Official PyTorch implementation of the NeurIPS 2020 paper

**Gonçalo M. Correia** (Instituto de Telecomunicações), **Vlad Niculae** (IvI, University of Amsterdam), **Wilker Aziz** (ILLC, University of Amsterdam), **André F. T. Martins** (Instituto de Telecomunicações, Unbabel, LUMLIS)


**Abstract**:

_Training neural network models with discrete (categorical or structured) latent variables can be computationally challenging, due to the need for marginalization over large or combinatorial sets. To circumvent this issue, one typically resorts to sampling-based approximations of the true marginal, requiring noisy gradient estimators (e.g., score function estimator) or continuous relaxations with lower-variance reparameterized gradients (e.g., Gumbel-Softmax). In this paper, we propose a new training strategy which replaces these estimators by an exact yet efficient marginalization. To achieve this, we parameterize discrete distributions over latent assignments using differentiable sparse mappings: sparsemax and its structured counterparts. In effect, the support of these distributions is greatly reduced, which enables efficient marginalization. We report successful results in three tasks covering a range of latent variable modeling applications: a semisupervised deep generative model, a latent communication game, and a generative model with a bit-vector latent representation. In all cases, we obtain good performance while still achieving the practicality of sampling-based approximations._

## Resources

- [Paper](https://arxiv.org/abs/2007.01919) (arXiv)


All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

## Python requirements and installation

This code was tested on `Python 3.7.1`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen) repository to your home: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/deep-spin/lp-sparsemap) repository to your home, and follow the installation instructions found there
4. Install PyTorch: `pip install torch` (we used version 1.6.0)
5. Install the requirements: `pip install -r requirements.txt`
6. Install the `lvm-helpers` package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)

## Preparing dataset



## Running

**Training**:

To train a network, run:

```
python 
```

**Evaluating**:

To evaluate a trained network against one of the test sets, run:

```
python 
```

The evaluation results should match the paper.

## Acknowledgements

This work was partly funded by the European Research Council (ERC StG DeepSPIN 758969), by the P2020 project MAIA (contract 045909), and by the Fundação para a Ciência e Tecnologia through contract UIDB/50008/2020. This work also received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement 825299 (GoURMET).
