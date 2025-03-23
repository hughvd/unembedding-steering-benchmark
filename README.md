# Evaluating Activation Steering Methods in LLMs

## Problem Description

Large Language Models (LLMs) have demonstrated remarkable capabilities, but controlling their outputs through activation steering methods remains an active area of research. While various steering techniques have emerged for more abstract concepts (truthfulness, deception, refusal), there lacks a set of good baselines to evaluate whether these directions truly represent high-level abstract concepts or are just statistical aggregations of tokens in the unembedding space. Understanding this distinction is crucial for developing more robust and generalizable steering methods.

## Research Objectives

This research proposes to develop a systematic benchmark for evaluating steering methods, focusing on residual stream interventions through token unembedding. Using Gemma-2-9b as our primary experimental platform, we will:

1. Investigate whether direct manipulation of the residual stream through targeted semantic token unembeddings can effectively influence model behavior across multiple steering objectives.
2. Compare the steering vectors produced by various steering techniques (CAA, ActAdd, etc.) with the unembedding vectors to evaluate their similarity.
3. Study whether we can learn steering vectors orthogonal to the unembedding vectors and compare their performance to the unrestricted case.

## Related Work

The field of representation engineering [1] focuses on strategies to manipulate and interpret the internal activations of Large Language Models (LLMs) during inference, rather than altering their weights. Techniques such as sparse autoencoders [2] and contrastive activation addition (CAA) [3] have been developed to identify specific "vectors" within the model's activation space that can enable the modification of model behavior in real time.

Recent studies have highlighted limitations in the generalizability and robustness of these methods. Research indicates that certain steering techniques may not consistently influence model behavior as intended across diverse contexts [4]. Additionally, evaluations of behavioral steering interventions have revealed that methods like CAA can be brittle, raising concerns about their reliability [5].

## Importance

Understanding the distinction between high-level abstract concepts and mere statistical token aggregations in LLMs is important for developing robust and generalizable steering methods. Activation steering techniques have shown promise in guiding LLM behavior but without a systematic benchmark to evaluate these methods, it's challenging to determine whether the steering directions genuinely represent the intended abstract concepts or are simply manipulating token-level statistics.

Establishing such benchmarks is essential to ensure that interventions lead to meaningful and predictable changes in model behavior, thereby enhancing the reliability and safety of LLM applications.

## Proposed Approach

### Steering Experiments

#### 1. Standard Steering in Comparison to Unembedding Steering

- **Pick concept**: e.g., sentiment = positive
- **Gather a relevant dataset**:
  - Positive contexts that cause the model to produce high logit for positive words (e.g., reviews)
  - Non-positive (neutral) contexts that don't (same dataset yet non-positive sentiment)
- **Collect hidden states** at layer L
- **Train linear probe (or CAA vector)** to classify positive vs. neutral from these states. The learned weight vector C from that linear probe is a candidate for concept direction
- **W_pos**: Collect token unembeddings for words related to positive sentiment ('Positive', 'Good', 'Great', 'Amazing', etc.)
  - Take mean, PC2, SVD, etc. of these unembeddings
- **Compare C directions with the unembedding for positive tokens**:
  - Record the product C * W_pos
  - If C is nearly orthogonal to W_pos, that suggests they are not aligned
  - If C has a large product, then they are somewhat aligned
  - Both of these are compared to dot product with a random set of vectors
- **Compare interventional impacts of C with W_pos**:
  - Steer the model activations at layer L with C and record output text
  - Steer activations using W_pos at layer L and record output text
  - Evaluate using benchmark or LLM judge to determine positive sentiment expression

#### 2. Orthogonal to Unembedding Steering

- **Learn steering vector distinct from token embeddings** by using projection/orthonormalization
- **Evaluate** if you can learn a classifier C_perp that classifies positivity in hidden states as positive or non-positive sentiment while remaining epsilon orthogonal to W_pos
- **Steer model activations** C_perp into test samples to see if positive sentiment is expressed
- **Compare** using dot product C_perp to C

#### 3. Interpretation

- If removing overlap with W_pos kills the positivity concept classification or drastically changes the model behavior, then the concept is partly aligned with that unembedding direction
- If it makes no difference, it's evident that the concept can be expressed in other directions too (thus orthogonal to the direct unembedding of "good")

### Potential Caveats

- It could be impossible to learn an orthogonal to the unembedding set vector that effectively steers the model
- By measuring the effect of adding C vs. W_pos we expect to reveal whether the learned concept direction is functionally similar to injecting positive token unembeddings. Constructing C_perp reveals that the concept is not solely aligned with the unembedding vector

## Expected Outcomes

- There will be some behaviors for which we can learn orthogonal vectors
- We expect behaviors like truthfulness and deception to have representations that are not aligned with token unembeddings
- On the other hand, we expect something like toxicity to be highly correlated with the embedding tokens
- We expect unembedding steering to work very well
  - As in previous work such as "A Mechanistic Case Study in Alignment Algorithm," a probe used to steer model behavior is trained on the last layer. This means it likely is picking up on the toxic unembeddings. Since the steering works really well in this case, we also expect unembedding steering to work well

## References

[1] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

[2] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Chris Olah. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. https://transformer-circuits.pub/2023/monosemantic-features.

[3] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition. arXiv e-prints, pages arXiv–2312, 2023.

[4]  Daniel Chee Hian Tan, David Chanin, Aengus Lynch, Adrià Garriga-Alonso, Dimitrios
Kanoulas, Brooks Paige, and Robert Kirk. Analyzing the generalization and reliability of
steering vectors. In ICML 2024 Workshop on Mechanistic Interpretability, 2024.

[5] Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, and Rada Mihalcea. A mechanistic understanding of alignment algorithms: A case study on DPO and toxicity. In Forty-first International Conference on Machine Learning, 2024.