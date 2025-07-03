# CS & applied-math-papers
Latest CS & math papers with a focus on ML/AI and some supplemental resources. Curated with a focus on 

1. Building foundational knowledge for the latest research topics
2. Understanding frontier research papers
3. Identifying limitations and future research directions

Created by Nicole Hao, 2025

## Neural Tangent Kernels (NTKs)

Start with the following:

- [Lilian Weng's Blog on Mathe Behind NTKs](https://lilianweng.github.io/posts/2022-09-08-ntk/)
- [Understanding the Neural Tangent Kernel](https://www.eigentales.com/NTK/)
- [Neural Tangent Kernel, Applied Probability Notes](https://appliedprobability.blog/2021/03/10/neural-tangent-kernel/)
- [Prior's for Infinite Networks](https://link.springer.com/chapter/10.1007/978-1-4612-0745-0_2)

If you're interested in the functional analysis foundations of NTKs
- [Similarity of Neural Network Models: A Survey of Functional and Representational Measures](https://arxiv.org/pdf/2305.06329)

Now you should be ready for the NTK foundational papers:
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572), NeruiPS 2018
- [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165), ICLR 2018
- [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)




## O(3)-Equivariant Deep Networks

O(3) is the group of all rotations and reflections in 3D space. It's a mathematical concept that describes how objects can be moved in 3D without changing their fundamental properties (like distances between points).

An equivariant neural network, on the other hand, is one where the output changes in a specific, predictable way when the input is transformed by a group operation (like rotation or reflection). In simpler terms, if you rotate the input, the output will also be rotated in a corresponding manner.

Why are O(3) equivariant networks important? This is because they can learn generalizable features from 3D data, even if the training data only contains examples in a limited number of orientations. This reduces the amount of training data needed.
Also, for tasks involving 3D objects, like predicting molecular properties or simulating physical systems, O(3)-equivariance ensures that the model's predictions are consistent with the laws of physics and geometry. They are also better at generalizing to unseen orientations of the input data.


Mathematical foundation on 3D rotation groups (group theory, lin alg, specifically rotation matrices) :
- [3D Rotation Grouo](https://en.wikipedia.org/wiki/3D_rotation_group), a 3D rotation group is a type of orthogonal group.
- [Orthogonal Group](https://en.wikipedia.org/wiki/Orthogonal_group)
- [3D Rotations](http://mesh.brown.edu/rotations/), Gabriel Taubin


Latest papers:
- [Unifying O(3) Equivariant Neural Networks Design with Tensor-Network Formalism](https://arxiv.org/abs/2211.07482)
- [An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks](https://arxiv.org/abs/2501.13986)

## Real-time adaptation laws for neural networks

Mathematical Foundations
- A college course in Differential equations. 
- Intermediate ODEs textbook, [Arnold's Ordinary Differential Equations](https://loshijosdelagrange.wordpress.com/wp-content/uploads/2013/04/vladimir-i-arnold-vladimir-i-arnold-roger-cooke-ordinary-differential-equations-1992.pdf)
- Dynamical systems
- Nonlinear dynmaics is helpful too. I recommend Prof. Strogatz's [Nonlinear dynamics and Chaos](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)

Concept definition more pertinent to understanding latest research: 
- [Lyapunov function](https://en.wikipedia.org/wiki/Lyapunov_function)
- [Lyapunov stability](https://en.wikipedia.org/wiki/Lyapunov_stability)

Latest papers:
- [A Tutorial on a Lyapunov-Based Approach to the Analysis of Iterative Optimization Algorithms](https://arxiv.org/abs/2309.11377)
- [Lyapunov-Based Real-Time and Iterative Adjustment of Deep Neural Networks](https://ieeexplore.ieee.org/document/9337905), IEEE Control Systems Letters


## Model Compression Techniques through numerical linear algebra
Mathematical foundation
- Linear algebra
- [Numerical analysis, numerical linear algebra, Trefethen and Bau](https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf), Chapter I - V

Latest papers:
- [Model Preserving Compression for Neural Networks](https://arxiv.org/abs/2108.00065), NeurIPS

## Graph Neural Networks (GNNs)
WIP

## Physics-informed Neural Networks (PINNs)
WIP



