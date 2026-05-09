# CS & Applied Math Papers

Curated ML/AI research papers with supplemental learning resources, organized around three goals: building foundational knowledge, understanding frontier research, and identifying limitations and future directions.

*Created by Nicole Hao, 2025*

---

## Contents

- [AI Foundations — Must Reads](#ai-foundations--must-reads)
- [Graph Neural Networks (GNNs)](#graph-neural-networks-gnns)
- [AI for Health (AI4Health)](#ai-for-health-ai4health)
- [Agentic AI](#agentic-ai)
- [AI for Science (AI4S)](#ai-for-science-ai4s)
  - [Physics — PINNs](#physics--physics-informed-neural-networks-pinns)
- [Embodied AI](#embodied-ai)
- [Artificial General Intelligence (AGI)](#artificial-general-intelligence-agi)
- [Efficient AI Model Architecture](#efficient-AI)
- [Uncategorized Topics](#uncategorized-topics)
  - [Liquid Neural Networks (LNNs)](#liquid-neural-networks-lnns)
  - [Neural Tangent Kernels (NTKs)](#neural-tangent-kernels-ntks)
  - [O(3)-Equivariant Deep Networks](#o3-equivariant-deep-networks)
  - [Real-time Adaptation Laws](#real-time-adaptation-laws-for-neural-networks)
  - [Model Compression via Numerical Linear Algebra](#model-compression-via-numerical-linear-algebra)

---

## AI Foundations — Must Reads

Landmark papers that every ML practitioner should know. Sorted by citation count.

| Paper | Year | Citations | Why it matters |
|---|---|---|---|
| [ResNet](https://arxiv.org/abs/1512.03385) | 2016 | 297k | Solved the challenge of training very deep networks via residual connections |
| [Adam](https://arxiv.org/abs/1412.6980) | 2014 | 200k | The default optimizer for nearly all large model training |
| [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 2012 | 187k | Sparked the modern deep learning era; demonstrated GPU-based training at scale |
| [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) | 2017 | 173k | The architecture underlying all modern LLMs |
| [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) | 1997 | 140k | Dominated NLP sequence modeling for 20 years |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | 120k | Introduced the pretraining + fine-tuning paradigm |
| [Deep Learning (review)](https://www.nature.com/articles/nature14539) | 2015 | 106k | LeCun, Hinton, Bengio survey — textbook-level status |
| [GAN](https://arxiv.org/abs/1406.2661) | 2014 | 105k | Generative adversarial networks; foundational for image generation |
| [VGG](https://arxiv.org/abs/1409.1556) | 2014 | 99k | Defined the design principles of deep vision networks |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | 2015 | 99k | Milestone in object detection; widely deployed in industry |
| [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | 1998 | 82k | The original CNN; foundational work by LeCun |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | 70k | Makes large-model training ~10× faster |
| [U-Net](https://arxiv.org/abs/1505.04597) | 2015 | 70k | Architectural foundation for segmentation and diffusion models |
| [t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html) | 2008 | 63k | Standard dimensionality reduction and data visualization technique |
| [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) | 2014 | 60k | The simplest effective regularization method for neural networks |

---

## Graph Neural Networks (GNNs)

Neural networks designed for graph-structured data. Nodes iteratively aggregate information from neighbors via **message passing**, producing learned **embeddings** that encode both node features and relational structure. Unlike CNNs or RNNs, GNNs operate directly on irregular, non-Euclidean data.

### Foundational Knowledge

- [Graph Neural Network — Wikipedia](https://en.wikipedia.org/wiki/Graph_neural_network) — Overview of building blocks: permutation equivariant layers, local/global pooling, and expressive power
- [Graph Neural Networks in TensorFlow — Google Research](https://research.google/blog/graph-neural-networks-in-tensorflow/) — Training on large datasets via subgraph streaming; supervised and unsupervised methods
- [GNNs: A New Frontier — NumberAnalytics](https://www.numberanalytics.com/blog/graph-neural-networks-new-frontier-cognitive-science-computing) — Core components (node representation, message passing, aggregation) with comparisons to traditional NNs

### Frontier Research

- [GNNs: Foundation, Frontiers and Applications](https://scholars.duke.edu/individual/pub1550511) — Tutorial covering fundamentals, new research directions, and applications in recommender systems and computer vision
- [Recent Research Progress of GNNs in Computer Vision](https://www.mdpi.com/2079-9292/14/9/1742) — Comprehensive review of image processing, video analysis, and multimodal data fusion

---

## AI for Health (AI4Health)

Applications of ML to healthcare: medical imaging, personalized treatment, drug discovery, and disease prediction. The field aims to improve diagnostic accuracy, enhance patient outcomes, and streamline clinical workflows.

### Foundational Knowledge

- [AI In Health and Health Care: Priorities for Action](https://www.healthaffairs.org/doi/10.1377/hlthaff.2024.01003) — *Health Affairs* — Historical context and policy priorities; from symbolic AI to modern deep networks for imaging and diagnostic reasoning
- [Impact of AI on Healthcare: A Review](https://www.researchgate.net/publication/372960293_Impact_of_Artificial_Intelligence_on_Healthcare_A_Review_of_Current_Applications_and_Future_Possibilities) — ML and NLP in imaging, diagnosis, treatment planning, and personalized medicine

### Frontier Research

- [Evolution of AI in Healthcare: A 30-Year Bibliometric Study](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1505692/full) — *Frontiers in Medicine* — Longitudinal analysis; highlights rise of COVID-19 analysis and drug discovery
- [Frontiers on Healthcare Research (journal)](https://frontiersonhealthcare.org/) — Bridges research and clinical application; peer-reviewed findings on patient care strategies

### Limitations & Open Directions

**Data dependency** — Models trained on single-institution data often fail to generalize across populations. Key need: diverse, multi-site training datasets and more robust model architectures.

**Bias & ethics** — Black-box models can exacerbate healthcare disparities and erode clinician trust. Key need: explainable AI (XAI) methods that make decision-making transparent.

**Clinical integration** — Many promising models have never been tested in real clinical settings. Key need: workflow-native deployment and proper training for healthcare professionals.

**Human-in-the-loop** — AI should assist, not replace, clinicians. Key need: systems where the final decision remains with a human, supported by transparent model reasoning.

---

## Agentic AI

> *Section in progress — more papers being added.*

### Reviews & History

- [Adaptation of Agentic AI](https://arxiv.org/abs/2512.16301) — arXiv 2025
- [Agentic AI: The Age of Reasoning — A Review](https://www.sciencedirect.com/science/article/pii/S2949855425000516) — ScienceDirect 2025

---

## AI for Science (AI4S)

AI applications across scientific disciplines. Papers have varying levels of domain specificity and are organized by field (AI4Physics, AI4Chemistry, etc.).

Reference collection: [AI4QC GitHub repo](https://github.com/AI4QC/AI_for_Science_paper_collection)

### Field Overviews

- [SAIBench: Structural Interpretation of AI for Science Through Benchmarks](https://arxiv.org/abs/2311.17869)
- [Bridging AI and Science: Implications from a Large-Scale Literature Analysis](https://arxiv.org/abs/2412.09628)

### Physics — Physics-Informed Neural Networks (PINNs)

PINNs embed governing equations (e.g. PDEs) directly into the loss function, enabling solutions to physical problems with far less training data than purely data-driven approaches. Useful for both forward problems (solving known equations) and inverse problems (discovering unknown parameters).

#### Start Here

- [Physics-Informed Neural Networks — Wikipedia](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) — Defines PINNs, explains function approximation, covers data-driven solution and discovery of PDEs
- [PINNs for Inverse PDE Problems — Towards Data Science](https://towardsdatascience.com/physics-informed-neural-networks-for-inverse-pde-problems/) — Intuitive explanation of automatic differentiation and physics-based loss functions
- [Adaptive PINNs (review) — arXiv 2025](https://arxiv.org/html/2503.18181v1) — Integrating transfer learning and meta-learning into PINNs to improve adaptivity and convergence

#### Frontier Papers

- [PINNs: Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers](https://www.mdpi.com/2076-3417/15/14/8092) — *MDPI Applied Sciences* — Comprehensive review; proposes roadmap toward "PINN 2.0" including neuro-symbolic integration and quantum-accelerated optimization
- [PINNs with Hard Constraints for Inverse Design](https://epubs.siam.org/doi/10.1137/21M1397908) — *SIAM* — Topology optimization via hard-constrained PINNs; significant advancement for engineering design

#### Known Limitations

- **Computational cost** — Multi-physics and high-dimensional problems are expensive to train
- **Optimization difficulty** — Convergence properties are less understood than classical numerical methods
- **High-frequency & multiscale problems** — PINNs struggle with high-frequency solutions and noisy, sparse real-world data
- **Future directions** — Domain decomposition (e.g. XPINNs), federated physics learning, quantum-accelerated optimization

---

## Embodied AI

> *Section in progress — more papers being added.*

Integrating AI into physical systems — robots, autonomous vehicles, smart machines — to perceive, reason, and act in the real world. Bridges computation and physical interaction via ML, computer vision, and sensor fusion.

### Introduction & Overviews

- [Toward Embodied AGI: A Review of Embodied AI and the Road Ahead](https://arxiv.org/pdf/2505.14235) — arXiv 2025

### Theory of Mind Inference

- [MMToM-QA: Multimodal Theory of Mind Question Answering](https://arxiv.org/abs/2401.08743) — arXiv 2024

---

## Artificial General Intelligence (AGI)

> *Section in progress — more papers being added.*

- [How Far Are We From AGI: Are LLMs All We Need?](https://arxiv.org/abs/2405.10313) — arXiv 2024



---

# Efficient AI Model Architecture


## Understanding the Landscape

The field broadly splits into two problems with different constraints:

**Training efficiency**: reducing the compute and energy cost of producing a model from scratch. Key levers: optimizer design, architecture search, sparsity during training, and scaling laws (knowing when to stop).

**Inference efficiency**: reducing cost per query at deployment. Often more commercially urgent since a model may run billions of inferences. Key levers: quantization, pruning, distillation, speculative decoding, and hardware-aware architecture design.

Many techniques help both, but the dominant bottleneck differs. Training is dominated by matrix multiplications on dense tensors; inference adds memory bandwidth as a bottleneck, loading weights from DRAM is often the limiter, not raw compute.

---

## Main Research Threads

### Architecture Design

The standard Transformer is not efficient by design, it was optimized for quality. 

Much active work targets the attention mechanism, which scales quadratically with sequence length.

**Sparse / linear attention**
- [Longformer](https://arxiv.org/abs/2004.05150): sliding window + global attention for long sequences
- [Performer](https://arxiv.org/abs/2009.14794): approximates attention with random feature maps
- [FlashAttention](https://arxiv.org/abs/2205.14135): hardware-aware exact attention; 2–4× faster with no quality loss; practically very important

**State space models (SSMs)**
- [Mamba](https://arxiv.org/abs/2312.00752): current frontier; linear-time sequence modeling as an alternative to attention entirely

**Mixture of experts (MoE)**
- Only activate a subset of parameters per token
- [Switch Transformer](https://arxiv.org/abs/2101.03961): foundational MoE paper; most frontier models now use this design

**Hybrid architectures**
- [Jamba](https://arxiv.org/abs/2403.19887): combines attention with SSMs; representative of current hybrid direction

### Training Efficiency

**Scaling laws**
- [Chinchilla](https://arxiv.org/abs/2203.15556) — showed most models are undertrained relative to their size; understanding optimal compute allocation is still an open problem

**Efficient optimizers**
- [Sophia](https://arxiv.org/abs/2305.14342): second-order optimizer that converges faster than Adam
- [SOAP](https://arxiv.org/abs/2409.11321): recent entry in efficient second-order methods

**Systems-level**: gradient checkpointing, mixed precision training, and distributed training strategies directly affect energy per training run; more engineering-oriented but high practical impact

### Model Compression (Post-training)

**Quantization**: reducing weight precision (fp32 → fp16 → int8 → int4)
- [GPTQ](https://arxiv.org/abs/2210.17323): post-training quantization for large language models
- [AWQ](https://arxiv.org/abs/2306.00978): activation-aware weight quantization

**Pruning** — removing weights or attention heads; structured pruning is more hardware-friendly than unstructured

**Knowledge distillation**: training a small student model to mimic a large teacher
- [DistilBERT](https://arxiv.org/abs/1910.01108): the canonical example


### Hardware-Aware Design

An underexplored angle for theory-oriented researchers. Most architectures are designed without explicit knowledge of memory hierarchies. Papers like FlashAttention demonstrate how hardware-aware kernel design can give 2–4× speedups with no quality loss.

- [FlashAttention-2](https://arxiv.org/abs/2307.08691) improved tiling and parallelism
- [Efficient GPU Kernels for N:M Sparse Weights](https://arxiv.org/abs/2104.02452) structured sparsity on modern GPUs


### Other Readings

Read these six in order before deciding on an angle.

1. **[Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02629)** Strubell et al., 2019 — Establishes the problem; read this first
2. **[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)** Foundational for training efficiency thinking
3. **[FlashAttention](https://arxiv.org/abs/2205.14135)** Best example of hardware-aware architecture work
4. **[Mamba](https://arxiv.org/abs/2312.00752)** Current frontier for efficient sequence modeling
5. **[A Survey of Efficient Transformers](https://arxiv.org/abs/2009.06732)** Good map of the attention efficiency space
6. **[GPTQ](https://arxiv.org/abs/2210.17323)** Representative inference quantization paper

---

## Uncategorized Topics

### Liquid Neural Networks (LNNs)

Biologically-inspired dynamic architectures that continuously learn and adapt in real-time, even after deployment — unlike standard fixed-weight networks.

- [Comparative Analysis: LNNs vs RNNs](https://www.researchgate.net/publication/392909174_Comparative_Analysis_Between_Liquid_Neural_Networks_LNNs_and_Recurrent_Neural_Networks_RNNs) — ResearchGate
- [Liquid Neural Networks for Telecom: From First Principles](https://arxiv.org/abs/2504.02352) — arXiv 2025

---

### Neural Tangent Kernels (NTKs)

#### Start Here

- [Math Behind NTKs — Lilian Weng](https://lilianweng.github.io/posts/2022-09-08-ntk/)
- [Understanding the NTK — eigentales.com](https://www.eigentales.com/NTK/)
- [NTK — Applied Probability Notes](https://appliedprobability.blog/2021/03/10/neural-tangent-kernel/)
- [Priors for Infinite Networks — Springer](https://link.springer.com/chapter/10.1007/978-1-4612-0745-0_2)
- [Similarity of NN Models: Functional and Representational Measures](https://arxiv.org/pdf/2305.06329) — for functional analysis foundations

#### Foundational Papers

- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) — NeurIPS 2018
- [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165) — ICLR 2018
- [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)

---

### O(3)-Equivariant Deep Networks

Networks where outputs transform predictably under 3D rotations and reflections (the O(3) group). Critical for molecular property prediction and physical simulation, where models must respect geometry. Reduces data requirements and improves generalization to unseen orientations.

#### Mathematical Foundations

- [3D Rotation Group — Wikipedia](https://en.wikipedia.org/wiki/3D_rotation_group)
- [Orthogonal Group — Wikipedia](https://en.wikipedia.org/wiki/Orthogonal_group)
- [3D Rotations — Gabriel Taubin, Brown University](http://mesh.brown.edu/rotations/)

#### Papers

- [Unifying O(3) Equivariant NN Design with Tensor-Network Formalism](https://arxiv.org/abs/2211.07482) — arXiv 2022
- [An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks](https://arxiv.org/abs/2501.13986) — arXiv 2025

---

### Real-time Adaptation Laws for Neural Networks

> **Prerequisites:** differential equations, intermediate ODEs, dynamical systems, nonlinear dynamics
>
> Recommended texts: [Arnold's Ordinary Differential Equations](https://loshijosdelagrange.wordpress.com/wp-content/uploads/2013/04/vladimir-i-arnold-vladimir-i-arnold-roger-cooke-ordinary-differential-equations-1992.pdf) · [Strogatz — Nonlinear Dynamics and Chaos](https://www.stevenstrogatz.com/books/nonlinear-dynamics-and-chaos-with-applications-to-physics-biology-chemistry-and-engineering)
>
> Key concepts: [Lyapunov function](https://en.wikipedia.org/wiki/Lyapunov_function) · [Lyapunov stability](https://en.wikipedia.org/wiki/Lyapunov_stability)

- [A Tutorial on a Lyapunov-Based Approach to Iterative Optimization Algorithms](https://arxiv.org/abs/2309.11377) — arXiv 2023
- [Lyapunov-Based Real-Time and Iterative Adjustment of Deep Neural Networks](https://ieeexplore.ieee.org/document/9337905) — IEEE Control Systems Letters

---

### Model Compression via Numerical Linear Algebra

> **Prerequisites:** linear algebra
>
> Recommended text: [Numerical Linear Algebra — Trefethen & Bau](https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf), chapters I–V

- [Model Preserving Compression for Neural Networks](https://arxiv.org/abs/2108.00065) — NeurIPS

---

*Target audience: graduate students in ML/AI, researchers entering new subfields, practitioners building research foundations.*