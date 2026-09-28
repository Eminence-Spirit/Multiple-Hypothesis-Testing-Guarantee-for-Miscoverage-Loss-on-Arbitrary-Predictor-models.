# Multiple Hypothesis Testing Guarantee for Miscoverage Loss on Arbitrary Predictor models.
This repository provides a presentation of how a high probability guarantee can be conferred on the population level miscoverage loss. It contains the explanation, the code and visualizations.


# Introduction and Motivation

Modern machine learning models can be trained to make accurate predictions, but their performance on future, unseen data is inherently uncertain. In many applications, it is therefore important not only to produce predictions, but also to provide statistical guarantees on the risk of those predictions being incorrect. Given an arbitrary trained predictor and finite data, how can we construct a prediction set whose population miscoverage risk is controlled at a desired level with high probability? The procedure should be applicable to arbitrary predictor models, without requiring assumptions about the specific model class used to generate the predictions.

A set around the model prediction miscovers when the true response falls outside the prediction set produced for the model's predictions. The miscoverage probability is therefore the probability that a future observation is not covered by the model's prediction set. In this project, we consider miscoverage where the response space is ordered real numbers that can be of any dimension. 
# Set up
Let $T=\{(X_1,Y_1),\ldots,(X_n,Y_n)\}$ denote the training data, and let $Z=\{(X_{n+1},Y_{n+1}),\ldots,(X_{n+c},Y_{n+c})\}$ denote the held-out set, where each $X_i \in \mathbb{R}^m$ represents an $m$-dimensional feature vector and each $Y_i \in \mathbb{R}^k$ represents the corresponding response. Suppose $\hat{f}$ is an arbitrary trained predictor model. For a future observation $(X,Y)$, where $X \in \mathbb{R}^m$ and $Y \in \mathbb{R}^k$, we would like the prediction set to contain the true response with high probability.

We want to control the population miscoverage risk at a desired level $\epsilon$:
```math
\mathbb{P}\left(Y \notin C(\hat{f}(X))\right)
\leq \epsilon.
```
Note that when the response space is one-dimensional and the prediction set is given by

```math
C(\hat{f}(X)) = \left(\hat{f}(X)-a,\;\hat{f}(X)+a\right),
```
the miscoverage probability can be written as

```math
\mathbb{P}\left(\left|Y-\hat{f}(X)\right|>a\right) \leq \epsilon.
```

Consider the two following situations. The first is when we have a perfect predictor function, the second is when the predictor function is somewhat inaccurate. The method will work on both, in the sense that , but it will give different predictor sets, depending on which function was choosen.

Perfect predictor:

The challenge is that the population risk is unknown, since we only have access to a finite amount of observed data. Therefore, our objective is not simply to estimate the miscoverage risk, but to construct a procedure that provides a probabilistic guarantee that the population risk is below the desired level.

In particular, we seek a procedure such that, with probability at least $1-\delta$,
```math
$$
\mathbb{P}\left(
R(\hat{f},C) \leq \epsilon
\right)
\geq 1-\delta,
$$
```
where $\delta$ represents the allowed probability that the guarantee fails.



# Hypothesis Setup
To obtain the desired guarantee on the population miscoverage loss, we formulate the problem as a multiple hypothesis testing problem.

To achieve this, suppose that we have a collection of candidate prediction sets
```math
C_1(\hat{f}(X)), C_2(\hat{f}(X)), \ldots, C_m(\hat{f}(X)).
```
For each candidate prediction set, let $\theta_i$ denote its population miscoverage probability:

```math
\theta_i
=
\mathbb{P}\left(Y \notin C_i(\hat{f}(X))\right).
```

We can then formulate the hypothesis test for each candidate prediction set as

```math
H_{0,i}:\quad \theta_i > \epsilon
\qquad \text{vs.} \qquad
H_{a,i}:\quad \theta_i \leq \epsilon,
\qquad i=1,\ldots,m.
```
Each hypothesis represents the statement that the corresponding prediction set fails to achieve the desired population miscoverage level.

The objective is to identify which hypotheses can be rejected. Rejecting $H_i$ provides evidence that the corresponding prediction set satisfies the desired miscoverage constraint:
```math
\mathbb{P}\left(Y \notin C_i(\hat{f}(X))\right) \leq \epsilon.
```
Since the hypotheses are considered simultaneously, testing each one independently at level $\alpha$ does not generally provide the desired overall error control. Multiple hypothesis testing allows us to account for this multiplicity by constructing $p$-values for the hypotheses $H_1,\ldots,H_m$ and applying a multiple testing procedure.

Thus, the problem of obtaining a high-probability guarantee on population miscoverage is transformed into a multiple hypothesis testing problem: by controlling the appropriate multiple-testing error rate, we can identify prediction sets for which the desired population miscoverage guarantee holds with high probability.
