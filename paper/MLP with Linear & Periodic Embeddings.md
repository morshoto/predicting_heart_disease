Hi everyone,

Recently, strong Neural Network architectures for tabular data, such as **TabM** and **RealMLP**, have been achieving impressive results. While these models are sophisticated, one fundamental component contributing to their success is how they handle **Numerical Features**.

Instead of feeding raw scalar values directly into the network—as is common in standard MLP baselines—these architectures often employ **Numerical Embeddings**.

To better understand this specific "building block" of modern Tabular DL, I created a notebook to implement these embedding techniques from scratch using PyTorch and benchmark them.

### Notebook Link

**[Experiments with Numerical Embeddings (Linear & Periodic)](https://www.kaggle.com/code/masayakawamata/mlp-numerical-embeddings?scriptVersionId=295754877)**

### What I Explored

I implemented three methods and evaluated them using 3-fold Stratified K-Fold:

**1. Baseline (Scalar Input)**
Standard approach. Numerical features are scaled and concatenated directly.

**2. Linear Embeddings**
Each numerical feature x is projected into a vector space using independent learnable weights.
$$\mathbf{e} = x \cdot \mathbf{W} + \mathbf{b}$$

**3. Periodic Embeddings**
The technique highlighted in recent research (NeurIPS 2022). It transforms scalar values using Sine and Cosine functions before projection to handle the "Spectral Bias" of neural networks.
$$\mathbf{e} = \text{Concat} \left[ \cos(2\pi x \mathbf{c} + \mathbf{b}), \sin(2\pi x \mathbf{c} + \mathbf{b}) \right]$$

### Results & Insights

Here are the results from the experiment:

| Method                 | Overall AUC  | Improvement   |
| :--------------------- | :----------- | :------------ |
| **Baseline (Scalar)**  | 0.953078     | -             |
| **Linear Embedding**   | 0.953104     | +0.000027     |
| **Periodic Embedding** | **0.954387** | **+0.001309** |

**My Takeaways:**

- **Linear Embedding** showed almost no improvement over the baseline in this simple setup. Just adding parameters didn't help much.
- **Periodic Embedding**, however, showed a **significant boost (+0.0013)**. This strongly suggests that transforming numerical inputs into periodic representations helps the MLP capture high-frequency patterns that are otherwise invisible to standard ReLU networks.

It seems that for tabular data, we cannot ignore Numerical Embeddings if we want to squeeze the most out of Deep Learning models.

---

### Further Room for Experiments

The reference paper mentions even more advanced techniques that are worth exploring:

- **Piecewise Linear Encoding (PLE):** Instead of Sin/Cos, this method uses bins (similar to histograms) and interpolates values. The paper suggests this can be even more effective than Periodic embeddings on certain datasets.
- **Target-Aware Binning:** Using decision trees to determine the optimal bins for PLE, effectively bringing GBDT-like splitting logic into the Neural Network input layer.

There is still a lot of room for experimentation!

### Reference Paper

- **"On Embeddings for Numerical Features in Tabular Deep Learning"** (Gorishniy et al., NeurIPS 2022)
    - [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)

Have fun!

Hi everyone,

Recently, strong Neural Network architectures for tabular data, such as **TabM** and **RealMLP**, have been achieving impressive results. While these models are sophisticated, one fundamental component contributing to their success is how they handle **Numerical Features**.

Instead of feeding raw scalar values directly into the network—as is common in standard MLP baselines—these architectures often employ **Numerical Embeddings**.

To better understand this specific "building block" of modern Tabular DL, I created a notebook to implement these embedding techniques from scratch using PyTorch and benchmark them.

### Notebook Link

**[Experiments with Numerical Embeddings (Linear & Periodic)](https://www.kaggle.com/code/masayakawamata/mlp-numerical-embeddings?scriptVersionId=295754877)**

### What I Explored

I implemented three methods and evaluated them using 3-fold Stratified K-Fold:

**1. Baseline (Scalar Input)**
Standard approach. Numerical features are scaled and concatenated directly.

**2. Linear Embeddings**
Each numerical feature x is projected into a vector space using independent learnable weights.
$$\mathbf{e} = x \cdot \mathbf{W} + \mathbf{b}$$

**3. Periodic Embeddings**
The technique highlighted in recent research (NeurIPS 2022). It transforms scalar values using Sine and Cosine functions before projection to handle the "Spectral Bias" of neural networks.
$$\mathbf{e} = \text{Concat} \left[ \cos(2\pi x \mathbf{c} + \mathbf{b}), \sin(2\pi x \mathbf{c} + \mathbf{b}) \right]$$

### Results & Insights

Here are the results from the experiment:

| Method                 | Overall AUC  | Improvement   |
| :--------------------- | :----------- | :------------ |
| **Baseline (Scalar)**  | 0.953078     | -             |
| **Linear Embedding**   | 0.953104     | +0.000027     |
| **Periodic Embedding** | **0.954387** | **+0.001309** |

**My Takeaways:**

- **Linear Embedding** showed almost no improvement over the baseline in this simple setup. Just adding parameters didn't help much.
- **Periodic Embedding**, however, showed a **significant boost (+0.0013)**. This strongly suggests that transforming numerical inputs into periodic representations helps the MLP capture high-frequency patterns that are otherwise invisible to standard ReLU networks.

It seems that for tabular data, we cannot ignore Numerical Embeddings if we want to squeeze the most out of Deep Learning models.

---

### Further Room for Experiments

The reference paper mentions even more advanced techniques that are worth exploring:

- **Piecewise Linear Encoding (PLE):** Instead of Sin/Cos, this method uses bins (similar to histograms) and interpolates values. The paper suggests this can be even more effective than Periodic embeddings on certain datasets.
- **Target-Aware Binning:** Using decision trees to determine the optimal bins for PLE, effectively bringing GBDT-like splitting logic into the Neural Network input layer.

There is still a lot of room for experimentation!

### Reference Paper

- **"On Embeddings for Numerical Features in Tabular Deep Learning"** (Gorishniy et al., NeurIPS 2022)
    - [arXiv:2203.05556](https://arxiv.org/abs/2203.05556)

Have fun!
