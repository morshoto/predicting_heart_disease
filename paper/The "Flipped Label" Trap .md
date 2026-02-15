Why scores are so compressed: _The "Flipped Label" Trap_

The reason the scores are so close is that we have likely hit the **Bayes Error Rate** of the dataset—the theoretical limit of predictability where features simply don't contain enough information to perfectly separate the classes.

**The Stats: Original vs. Synthetic.**
The rate of samples where the ground truth contradicts the clinical features is remarkably consistent:

- **Original Dataset**: 47 flipped / 270 samples (**17.41%**)
- **Synthetic (Train)**: 69,872 flipped / 630,000 samples (**11.09%**)

I did a deep dive into the "**flipped labels**" (samples where a well-tuned model is confidently wrong) and discovered this isn't a synthetic artifact; it's inherited clinical complexity.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F7192196%2Fe248c3a758a271e1fcfc2a6ed39254ae%2FScreenshot%202026-02-12%20094731.png?generation=1770886082411640&alt=media)

### Case Study: The 'Conditional Noise' (Flipped Presence)

I’ve highlighted **Row 70** and **Row 72** because they represent different **"flavors"** of the same problem. In both cases, the patient has heart disease (**y=1**), but a baseline model **"flips"** them to a healthy prediction.

---

#### **Example A: The Silent Case (Index 70)**

**Features:** Age **64** | Sex **0** | ST Dep. **0.0** | Vessels **0** | Thal **7**

> > **The Trap:** Having **0.0 ST depression** and **0 vessels** usually signal "Healthy." The model sees these strong indicators and assigns a very low-risk probability, but the ground truth is **1 (Sick)**. This patient is a "silent" high-risk case that contradicts standard clinical logic.

---

#### **Example B: The Contradictory Case (Index 72)**

> **Features:** Age **54** | Sex **1** | ST Dep. **1.2** | Vessels **1** | Thal **3**
>
> > **The Trap:** This patient has some risk markers (**ST 1.2** and **1 vessel**), but other features (like **Thal 3**) pull the model back toward a "Healthy" prediction. It’s a conflict of signals that the model fails to resolve correctly without deeper interactions.

**Why we are stuck:**

- The **0.9526 Baseline**: Standard models easily learn the "clean" logic (e.g., Low ST Depression = Healthy). This covers the ~89% of clear cases and gets you to 0.952.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F7192196%2F4655bceabb34af992674b6c02d5b2096%2Foutput.png?generation=1770884734904445&alt=media)

- **The "Perfect AUC" Illusion**: If you "clean" your data by removing these 70k flipped rows, your CV will hit 0.999, but your LB score will collapse. The test set still contains that 11% noise, and a model trained on "clean" data will be too overconfident to handle it.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F7192196%2F5040f307306be12f4eac851c03d9eaa4%2Foutput1.png?generation=1770884876479885&alt=media)

- **The 0.954+ Gap**: The top scores belong to those modeling the **conditional noise** and—crucially—those who **ensemble properly**. As [noted] (https://www.kaggle.com/competitions/playground-series-s6e2/discussion/672651) by @tilii7, the beauty of the AUC metric is that we don't need to "solve" these flips by pushing them across the 0.5 threshold. We just need to improve their relative ranking. Even moving a probability from 0.1 to 0.2—ranking it correctly against just a few other samples—pushes the AUC higher.
