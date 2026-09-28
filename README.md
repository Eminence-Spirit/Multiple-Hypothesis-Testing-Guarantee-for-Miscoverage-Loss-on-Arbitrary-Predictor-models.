# Multiple-Hypothesis-Testing-Guarantee-for-Miscoverage-Loss-on-Arbitrary-Predictor-models.
This repository provides a presentation of how a high probability guarantee can be conferred on the population level miscoverage loss. It contains the explanation, the code and visualizations.

#Motivation

Modern machine learning models can be trained to make accurate predictions, but their performance on future, unseen data is inherently uncertain. In many applications, it is therefore important not only to produce predictions, but also to provide statistical guarantees on the risk of those predictions being incorrect.

Our goal is to develop a general procedure that provides a high-probability guarantee on the population-level risk of an already-trained predictor model. The procedure should be applicable to arbitrary predictor models, without requiring assumptions about the specific model class used to generate the predictions.

In this project, we focus on miscoverage loss as the risk of interest. Suppose that an already-trained predictor $\hat{f}$ is used to construct a prediction set $C(\hat{f}(X))$, centered around the prediction made by $\hat{f}$. For a future observation $(X,Y)$, where $X$ represents the features and $Y$ represents the response, we would like the prediction set to contain the true response with high probability.

More precisely, we want to control the population miscoverage risk at a desired level $\epsilon$:

\mathbb{P}\left(Y \notin C(\hat{f}(X))\right)
\leq \epsilon.
$$

Here, the probability is taken over a future observation $(X,Y)$ from the underlying population. Importantly, $X$ and $Y$ can be arbitrarily dimensional, and $\hat{f}$ can be any trained predictor model.

The challenge is that the population risk is unknown, since we only have access to a finite amount of observed data. Therefore, our objective is not simply to estimate the miscoverage risk, but to construct a procedure that provides a probabilistic guarantee that the population risk is below the desired level.

In particular, we seek a procedure such that, with probability at least $1-\delta$,

$$
\mathbb{P}\left(
R(\hat{f},C) \leq \epsilon
\right)
\geq 1-\delta,
$$

where $\delta$ represents the allowed probability that the guarantee fails.

Thus, the central question addressed in this repository is:

Given an arbitrary trained predictor and finite data, how can we construct a prediction set whose population miscoverage risk is controlled at a desired level with high probability?

The approach developed in this project uses multiple hypothesis testing to obtain this guarantee while allowing the procedure to adapt to the characteristics of the data and the trained predictor.

# Hypothesis Setup

To obtain the desired guarantee on the population miscoverage risk, we formulate the problem as a multiple hypothesis testing problem.

Recall that our objective is to ensure

\mathbb{P}\left(Y \notin C(\hat{f}(X))\right)
\leq \epsilon.
$$

Suppose that the prediction set is constructed using a parameter $\lambda$, such that

$$
C_\lambda(\hat{f}(X))
$$

denotes the prediction set corresponding to $\lambda$. Increasing $\lambda$ can, for example, make the prediction set wider and therefore reduce its miscoverage probability.

For a given value of $\lambda$, we would like to determine whether the corresponding population miscoverage risk satisfies the desired bound $\epsilon$. This can be expressed through the hypothesis pair

$$
H_\lambda:
\quad
R(\hat{f}, C_\lambda) > \epsilon
$$

versus

$$
H_\lambda':
\quad
R(\hat{f}, C_\lambda) \leq \epsilon.
$$

However, rather than testing only a single value of $\lambda$, we consider a collection of candidate values

$$
\lambda_1,\lambda_2,\ldots,\lambda_m.
$$

This gives rise to a family of hypotheses

$$
H_j:
\quad
R(\hat{f}, C_{\lambda_j}) > \epsilon,
\qquad j=1,\ldots,m.
$$

Each hypothesis corresponds to the statement that the prediction set associated with $\lambda_j$ fails to achieve the desired population miscoverage level.

The goal is therefore to identify which candidate prediction sets satisfy the desired risk constraint while controlling the probability of making an incorrect selection.

Why Multiple Hypothesis Testing?

The candidate prediction sets are evaluated simultaneously. If each candidate were tested independently at level $\alpha$, the probability of making at least one incorrect decision across all candidates could be substantially larger than $\alpha$.

Multiple hypothesis testing provides a framework for controlling this multiplicity. In particular, we construct $p$-values for the hypotheses

$$
H_1,\ldots,H_m
$$

and apply a multiple hypothesis testing procedure to determine which hypotheses can be rejected while controlling an appropriate global error rate.

If a hypothesis

$$
H_j: R(\hat{f}, C_{\lambda_j}) > \epsilon
$$

is rejected, this provides evidence that the corresponding prediction set satisfies

$$
R(\hat{f}, C_{\lambda_j}) \leq \epsilon.
$$

Consequently, the set of rejected hypotheses identifies candidate prediction sets for which the desired population-level miscoverage guarantee can be established.

The key idea of this project is therefore to transform the problem of controlling population miscoverage risk into a multiple hypothesis testing problem, allowing established multiple testing procedures to provide the required high-probability guarantee.
