# Awesome Applied Math & AI Papers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Latest CS & math papers with a focus on ML/AI and some supplemental resources. Curated with a focus on:

1. Building foundational knowledge for the latest research topics
2. Understanding frontier research papers
3. Identifying limitations and future research directions

*Created by Nicole Hao, 2025*

---

## Table of Contents

- [AI Foundations](#ai-foundations-must-reads)
- [Graph Neural Networks (GNNs)](#graph-neural-networks-gnns)
- [State Space Models (SSMs) & Mamba](#state-space-models-ssms--mamba)
- [AI4Health](#ai4health)
- [Agentic AI](#agentic-ai)
- [AI for Science (AI4S)](#ai-for-science-ai4s)
  - [Physics — PINNs](#physics--physics-informed-neural-networks-pinns)
- [Embodied AI](#embodied-ai)
- [Artificial General Intelligence (AGI)](#artificial-general-intelligence-agi)
- [Liquid Neural Networks (LNNs)](#liquid-neural-networks-lnns)
- [Neural Tangent Kernels (NTKs)](#neural-tangent-kernels-ntks)
- [O(3)-Equivariant Deep Networks](#o3-equivariant-deep-networks)
- [Real-Time Adaptation Laws for Neural Networks](#real-time-adaptation-laws-for-neural-networks)
- [Model Compression via Numerical Linear Algebra](#model-compression-via-numerical-linear-algebra)

---

## AI Foundations, must reads

> Seminal papers that every ML/AI practitioner should know. Sorted by citation count.

| Paper | Venue | Citations | Why it matters |
|-------|-------|-----------|----------------|
| **[ResNet](https://arxiv.org/abs/1512.03385)** | CVPR 2016 | 297,000+ | 'Solved' the difficulty of training very deep networks |
| **[Adam](https://arxiv.org/abs/1412.6980)** | ICLR 2015 | 200,000+ | Most widely used optimizer; nearly all large models are trained with it |
| **[AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)** | NeurIPS 2012 | 187,000+ | Starting point of the deep learning boom; kicked off GPU-based training |
| **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** | NeurIPS 2017 | 173,000+ | The ancestor of all LLMs like ChatGPT |
| **[LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)** | Neural Computation 1997 | 140,000+ | Pioneer of sequence modeling; dominated NLP for 20 years |
| **[BERT](https://arxiv.org/abs/1810.04805)** | NAACL 2019 | 120,000+ | Introduced the "pretraining + fine-tuning" paradigm |
| **[Deep Learning (Review)](https://www.nature.com/articles/nature14539)** | Nature 2015 | 106,000+ | Review by LeCun, Hinton, Bengio; textbook-level status |
| **[GAN](https://arxiv.org/abs/1406.2661)** | NeurIPS 2014 | 105,000+ | Generative adversarial networks; early landmark for image generation |
| **[VGG](https://arxiv.org/abs/1409.1556)** | ICLR 2015 | 99,000+ | Classic CNN architecture; defined the design of deep vision networks |
| **[Faster R-CNN](https://arxiv.org/abs/1506.01497)** | NeurIPS 2015 | 99,000+ | Milestone in object detection; extremely widely used in industry |
| **[LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)** | IEEE 1998 | 82,000+ | The original CNN; foundational work for convolutional neural networks |
| **[Batch Normalization](https://arxiv.org/abs/1502.03167)** | ICML 2015 | 70,000+ | Makes large-model training ~10× faster |
| **[U-Net](https://arxiv.org/abs/1505.04597)** | MICCAI 2015 | 70,000+ | Architectural foundation for image segmentation and diffusion models |
| **[t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html)** | JMLR 2008 | 63,000+ | Must-read for dimensionality reduction and data visualization |
| **[Dropout](https://jmlr.org/papers/v15/srivastava14a.html)** | JMLR 2014 | 60,000+ | The simplest effective way to prevent neural network overfitting |

---

## Graph Neural Networks (GNNs)

> GNNs operate on irregular, graph-structured data using a **message-passing** framework — nodes update their representations by aggregating information from neighbors — producing learned **embeddings** of graph structure.

*Reference repo: [thunlp/GNNPapers](https://github.com/thunlp/GNNPapers)*

### Foundational Resources

- **[Graph Neural Networks — Wikipedia](https://en.wikipedia.org/wiki/Graph_neural_network)** — Overview of building blocks: permutation equivariant layers, local/global pooling, and expressive power of GNNs.
- **[Graph Neural Networks in TensorFlow](https://research.google/blog/graph-neural-networks-in-tensorflow/)** — Google Research: training GNNs on large datasets via subgraph streaming; supervised and unsupervised methods.
- **[Graph Neural Networks: A New Frontier](https://www.numberanalytics.com/blog/graph-neural-networks-new-frontier-cognitive-science-computing)** — Conceptual walkthrough of node representation, message passing, and aggregation, with comparisons to traditional NNs.

### Key Papers

- **[Graph Neural Networks: Foundation, Frontiers and Applications](https://scholars.duke.edu/individual/pub1550511)** — Broad tutorial covering fundamental concepts, research frontiers, and applications in recommender systems and computer vision.
- **[Recent Research Progress of GNNs in Computer Vision](https://www.mdpi.com/2079-9292/14/9/1742)** (Electronics, 2025) — Review of GNN applications in image processing, video analysis, and multimodal fusion.

---

## State Space Models (SSMs) & Mamba

> SSMs are a class of sequence models rooted in **control theory**, representing dynamics as linear recurrences: `h'(t) = Ah(t) + Bx(t)`, `y(t) = Ch(t)`. Unlike Transformers with O(n²) attention, SSMs process sequences in **O(n) time and O(1) recurrent state** during inference. **Mamba** (2023) introduced *input-selective* state spaces — the transition matrices A, B, C become functions of the input token, enabling content-aware filtering that prior fixed-transition SSMs (like S4) lacked.

### Foundational Resources

- **[S4: Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)** (ICLR 2022, Outstanding Paper) — Gu et al.; introduced the first practically efficient SSM using diagonal-plus-low-rank structure. The direct predecessor to Mamba.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)** (arXiv 2023) — Gu & Dao; the core Mamba paper. Introduces the S6 (selective scan) layer and a hardware-aware parallel scan algorithm. Start here for the architecture.
- **[The Annotated S4](https://srush.github.io/annotated-s4/)** — Sasha Rush; annotated implementation of S4, best hands-on introduction to SSM mechanics.

### Key Papers

- **[Understanding Input Selectivity in Mamba](https://machinelearning.apple.com/research/understanding-input)** (ICML 2025), Huang et al. — Mechanistic analysis of Mamba's S6 layer: shows it can represent Haar wavelet projections to approximate discontinuous functions, dynamically counteract memory decay, and analytically solve associative recall tasks.

- **[MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)** (CVPR 2025), Hatamizadeh & Kautz — Integrates self-attention blocks into the final stages of a Mamba backbone; captures long-range spatial dependencies that pure SSMs miss, achieving SOTA on ImageNet-1K and strong results on detection and segmentation.

- **[Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191)** (arXiv 2026), Moudgil et al. — Two-stage knowledge distillation from Transformer teachers into Mamba students using principled initialization and linearized attention; preserves teacher accuracy while retaining SSM inference efficiency.

### Limitations & Future Directions

- Input-selective scanning is powerful but harder to parallelize than attention on modern GPU hardware; hardware-aware implementations are still an active area.
- Mamba struggles with tasks requiring fine-grained, position-sensitive retrieval compared to full attention — motivating hybrid architectures like MambaVision.
- Theoretical understanding of *when* selectivity helps (vs. fixed transitions) remains incomplete; the ICML 2025 work above is an early step.
- Distilling Transformer knowledge into SSMs (cross-architecture distillation) is emerging as a practical path to SSM adoption without training from scratch.

---

## AI4Health

> Applying ML/AI to healthcare: medical image analysis, drug discovery, personalized treatment, and disease prediction.

### Foundational Resources

- **[AI In Health And Health Care: Priorities For Action](https://www.healthaffairs.org/doi/10.1377/hlthaff.2024.01003)** (Health Affairs, 2024) — Historical context and key policy domains; traces evolution from symbolic AI to modern deep neural networks.
- **[Impact of AI on Healthcare: Current Applications and Future Possibilities](https://www.researchgate.net/publication/372960293_Impact_of_Artificial_Intelligence_on_Healthcare_A_Review_of_Current_Applications_and_Future_Possibilities)** — Review of ML/NLP in image analysis, diagnosis, treatment planning, and personalized medicine.

### Key Papers

- **[Evolution of AI in Healthcare: A 30-Year Bibliometric Study](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1505692/full)** (Frontiers in Medicine, 2024) — Longitudinal analysis of AI publications; highlights growth in COVID-19 analysis and drug discovery.
- **[Frontiers on Healthcare Research](https://frontiersonhealthcare.org/)** — Peer-reviewed journal bridging healthcare research and clinical application.

### Limitations & Future Directions

- **Data dependency** — Models trained on single-institution data generalize poorly; robust, population-diverse models are a key open challenge.
- **Bias and ethics** — Training data biases can exacerbate disparities; "black box" models lack the transparency clinicians need to trust them.
- **Clinical integration** — Research performance rarely translates directly to clinical workflows; deployment requires careful re-evaluation and staff training.
- **Human-in-the-loop** — The near-term consensus is AI assists, not replaces, clinicians. Future work focuses on explainable AI (XAI) where final decisions remain with humans.

---

## Agentic AI

> AI systems that autonomously plan, reason, and act over extended horizons — often orchestrating tools and other models to complete complex tasks.

### Survey / History

- **[Adaptation of Agentic AI](https://arxiv.org/abs/2512.16301)** (arXiv, 2025)
- **[Agentic AI: The Age of Reasoning — A Review](https://www.sciencedirect.com/science/article/pii/S2949855425000516)** (2025)

*WIP — more papers coming.*

---

## AI for Science (AI4S)

> Applying AI to accelerate scientific discovery. Papers range from broad overviews to field-specific applications. Reference repo: [AI4QC/AI\_for\_Science\_paper\_collection](https://github.com/AI4QC/AI_for_Science_paper_collection).

### Overviews

- **[SAIBench: A Structural Interpretation of AI for Science Through Benchmarks](https://arxiv.org/abs/2311.17869)** (arXiv, 2023)
- **[Bridging AI and Science: Implications from a Large-Scale Literature Analysis of AI4Science](https://arxiv.org/abs/2412.09628)** (arXiv, 2024)

---

### Physics — Physics-Informed Neural Networks (PINNs)

> PINNs embed governing equations (e.g., PDEs) directly into the loss function, enabling solutions to physical problems with far less training data than purely data-driven models.

#### Foundational Resources

- **[Physics-Informed Neural Networks — Wikipedia](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)** — Defines PINNs, explains function approximation, and covers forward/inverse problem types.
- **[PINNs for Inverse PDE Problems](https://towardsdatascience.com/physics-informed-neural-networks-for-inverse-pde-problems/)** — Intuitive explanation of how automatic differentiation and physics-based loss functions outperform purely data-driven approaches.
- **[Adaptive Physics-Informed Neural Networks](https://arxiv.org/html/2503.18181v1)** (arXiv, 2025) — Review of transfer learning and meta-learning integration to improve PINN adaptivity and convergence.

#### Key Papers

- **[PINNs: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers](https://www.mdpi.com/2076-3417/15/14/8092)** (Applied Sciences, 2025) — Comprehensive review; proposes a "PINN 2.0" roadmap covering neuro-symbolic integration and quantum-accelerated optimization.
- **[PINNs with Hard Constraints for Inverse Design](https://epubs.siam.org/doi/10.1137/21M1397908)** (SIAM J. Scientific Computing, 2021) — Applies PINNs with hard constraints to topology optimization; significant advance for engineering design.

#### Limitations & Future Directions

- Computationally expensive for complex or multi-physics problems.
- Convergence properties less understood than classical numerical methods.
- Struggles with high-frequency and multiscale problems, and with noisy/sparse real-world data.
- Key research directions: domain decomposition (XPINNs), federated physics learning, quantum-accelerated optimization.

---

## Embodied AI

> Integrating AI into physical systems — robots, autonomous vehicles, industrial machines — to perceive, reason, and act in the real world.

### Introduction

- **[Toward Embodied AGI: A Review of Embodied AI and the Road Ahead](https://arxiv.org/pdf/2505.14235)** (arXiv, 2025)

### Theory of Mind

- **[MMToM-QA: Multimodal Theory of Mind Question Answering](https://arxiv.org/abs/2401.08743)** (arXiv, 2024)

*WIP — more papers coming.*

---

## Artificial General Intelligence (AGI)

- **[How Far Are We From AGI: Are LLMs All We Need?](https://arxiv.org/abs/2405.10313)** (arXiv, 2024)

*WIP — more papers coming.*

---

## Liquid Neural Networks (LNNs)

> LNNs use a dynamic architecture inspired by biological neurons, designed to continuously adapt in real-time after deployment.

- **[Comparative Analysis Between LNNs and RNNs](https://www.researchgate.net/publication/392909174_Comparative_Analysis_Between_Liquid_Neural_Networks_LNNs_and_Recurrent_Neural_Networks_RNNs)**
- **[Liquid Neural Networks: Next-Generation AI for Telecom from First Principles](https://arxiv.org/abs/2504.02352)** (arXiv, 2025)

---

## Neural Tangent Kernels (NTKs)

> NTKs describe the behavior of infinitely wide neural networks, connecting deep learning theory to kernel methods and Gaussian processes.

### Start Here

- **[The Math Behind NTKs](https://lilianweng.github.io/posts/2022-09-08-ntk/)** — Lilian Weng's blog; excellent starting point.
- **[Understanding the Neural Tangent Kernel](https://www.eigentales.com/NTK/)** — Eigentales blog, intuitive walkthrough.
- **[NTK — Applied Probability Notes](https://appliedprobability.blog/2021/03/10/neural-tangent-kernel/)**
- **[Priors for Infinite Networks](https://link.springer.com/chapter/10.1007/978-1-4612-0745-0_2)** — Neal (1996); connects infinite networks to Gaussian processes.

### Functional Analysis Foundations

- **[Similarity of Neural Network Models: A Survey of Functional and Representational Measures](https://arxiv.org/pdf/2305.06329)** (arXiv, 2023)

### Key Papers

- **[Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)** (NeurIPS 2018) — The foundational NTK paper.
- **[Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)** (ICLR 2018)
- **[On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)** (NeurIPS 2019)

---

## O(3)-Equivariant Deep Networks

> O(3) is the group of rotations and reflections in 3D space. Equivariant networks guarantee that rotating the input produces a correspondingly rotated output — critical for molecular property prediction and physical simulation.

**Why it matters:** Equivariance allows generalization across unseen orientations, reduces training data requirements, and ensures predictions are consistent with the laws of physics and geometry.

### Mathematical Foundations

- **[3D Rotation Group — Wikipedia](https://en.wikipedia.org/wiki/3D_rotation_group)** — Background on SO(3) and its relation to orthogonal groups.
- **[Orthogonal Group — Wikipedia](https://en.wikipedia.org/wiki/Orthogonal_group)**
- **[3D Rotations](http://mesh.brown.edu/rotations/)** — Gabriel Taubin; rotation matrices and representations.

### Key Papers

- **[Unifying O(3) Equivariant Neural Networks Design with Tensor-Network Formalism](https://arxiv.org/abs/2211.07482)** (arXiv, 2022)
- **[An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks](https://arxiv.org/abs/2501.13986)** (arXiv, 2025)

---

## Real-Time Adaptation Laws for Neural Networks

> Using Lyapunov stability theory and dynamical systems to derive update laws that guarantee stable, real-time learning.

### Mathematical Foundations

- Undergraduate ODEs course (prerequisite)
- **[Arnold's Ordinary Differential Equations](https://loshijosdelagrange.wordpress.com/wp-content/uploads/2013/04/vladimir-i-arnold-vladimir-i-arnold-roger-cooke-ordinary-differential-equations-1992.pdf)** — Intermediate ODE reference.
- Dynamical systems + nonlinear dynamics — **[Nonlinear Dynamics and Chaos](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)**, Strogatz.
- **[Lyapunov Function — Wikipedia](https://en.wikipedia.org/wiki/Lyapunov_function)**
- **[Lyapunov Stability — Wikipedia](https://en.wikipedia.org/wiki/Lyapunov_stability)**

### Key Papers

- **[A Tutorial on a Lyapunov-Based Approach to the Analysis of Iterative Optimization Algorithms](https://arxiv.org/abs/2309.11377)** (arXiv, 2023)
- **[Lyapunov-Based Real-Time and Iterative Adjustment of Deep Neural Networks](https://ieeexplore.ieee.org/document/9337905)** (IEEE Control Systems Letters, 2021)

---

## Model Compression via Numerical Linear Algebra

> Using matrix decomposition and approximation techniques (SVD, randomized NLA, etc.) to compress neural networks while preserving performance.

### Mathematical Foundations

- Linear algebra (prerequisite)
- **[Numerical Linear Algebra](https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf)** — Trefethen & Bau; Chapters I–V.

### Key Papers

- **[Model Preserving Compression for Neural Networks](https://arxiv.org/abs/2108.00065)** (NeurIPS 2021)
