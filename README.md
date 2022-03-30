## Part-based Pseudo Label Refinement (PPLR)

Official PyTorch implementation of [Part-based Pseudo Label Refinement for Unsupervised Person Re-identification](https://arxiv.org/abs/2203.14675) (CVPR 2022).
Code will be released soon.
Stay tuned.

## Overview
![overview](figs/overview.jpg)
>We propose a Part-based Pseudo Label Refinement (PPLR) framework that reduces the label noise by employing the complementary relationship between global and part features.
Specifically, we design a cross agreement score as the similarity of k-nearest neighbors between feature spaces to exploit the reliable complementary relationship. 
Based on the cross agreement, we refine pseudo-labels of global features by ensembling the predictions of part features, which collectively alleviate the noise in global feature clustering. 
We further refine pseudo-labels of part features by applying label smoothing according to the suitability of given labels for each part.
Our PPLR learns discriminative representations with rich local contexts. Also, it operates in a self-ensemble manner without auxiliary teacher networks, which is computationally efficient.

[//]: # (## Requirements)

[//]: # (## Training)

[//]: # (## Evaluataion)

[//]: # (## Citation)