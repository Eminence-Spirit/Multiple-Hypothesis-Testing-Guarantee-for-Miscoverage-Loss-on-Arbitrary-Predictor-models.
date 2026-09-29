# Multiple Hypothesis Testing Guarantee for Miscoverage Loss on Arbitrary Predictor models.
This repository provides a presentation of how a high probability guarantee can be conferred on the population level miscoverage loss. It contains the explanation, the code and visualizations.


# Introduction and Motivation

Modern machine learning models can be trained to make accurate predictions, but their performance on future, unseen data is inherently uncertain. In many applications, it is therefore important not only to produce predictions, but also to provide statistical guarantees on the risk of those predictions being incorrect. Given an arbitrary trained predictor and finite data, how can we construct a prediction set whose population miscoverage risk is controlled at a desired level with high probability? The procedure should be applicable to arbitrary predictor models, without requiring assumptions about the specific model class used to generate the predictions.

A set around the model prediction miscovers when the true response falls outside the prediction set produced for the model's predictions. The miscoverage probability is therefore the probability that a future observation is not covered by the model's prediction set. In this project, we consider miscoverage where the response space is ordered real numbers that can be of any dimension. 
# Set Up
Let $T=\{(X_1,Y_1),\ldots,(X_n,Y_n)\}$ denote the training data, and let $Z=\{(X_{n+1},Y_{n+1}),\ldots,(X_{n+c},Y_{n+c})\}$ denote the held-out set, where each $X_i \in \mathbb{R}^m$ represents an $m$-dimensional feature vector and each $Y_i \in \mathbb{R}^k$ represents the corresponding response. Suppose $\hat{f}$ is an arbitrary trained predictor model. For a given held-out set $Z$, let $C_Z(\hat{f}(X)) \subseteq \mathbb{R}^k$ denote a prediction set that depends on the held-out set $Z$ and the trained predictor $\hat{f}$, and is a subset of the response space. For a future observation $(X,Y)$, where $X \in \mathbb{R}^m$ and $Y \in \mathbb{R}^k$, we would like the prediction set to contain the true response with high probability. A note that enhances practicality of the technique related to conditionality of the claims is appended at the end of this document.*

We want to control the population miscoverage risk at a desired level $\epsilon$:
```math
\mathbb{P}_{(X,Y)}\left(
Y \notin C_Z(\hat{f}(X)) \;\Bigg|\; Z
\right)
\leq \epsilon.
```

The challenge is that the population risk is unknown, since we only have access to a finite amount of observed data. Therefore, our objective is not simply to estimate the miscoverage risk, but to construct a procedure that provides a probabilistic guarantee that the population risk is below the desired level.

In particular, we seek a procedure such that, with probability at least $1-\alpha$,
```math
\mathbb{P}_{Z}\left(
\mathbb{P}_{(X,Y)}\left(
Y \notin C_Z(\hat{f}(X)) \;\Bigg|\; Z
\right)
\leq \epsilon
\right)
\geq 1-\alpha.
```
where $\alpha$ represents the overall significance level.



Consider the two following situations, note that all the data points are resampled (not simply the one from the training or the calibration set) to provide a more accurate picture of the guarantee:
Suppose we have a perfect predictor (we were able to obtain the exact conditional mean), then we obtain the following:
![image alt](https://github.com/Eminence-Spirit/Multiple-Hypothesis-Testing-Guarantee-for-Miscoverage-Loss-on-Arbitrary-Predictor-models./blob/main/images_for_READ_ME/PERF_radius%20band.png?raw=true)

However, usually we are unable to obtain the perfect predictor, in which case, we can still apply the method, however, our band becomes larger to adapt to account for the predictor being imperfect. For instance, the following is obtained when the predictor is imperfect:
![image alt](https://github.com/Eminence-Spirit/Multiple-Hypothesis-Testing-Guarantee-for-Miscoverage-Loss-on-Arbitrary-Predictor-models./blob/main/images_for_READ_ME/Imperfect_radius_band.png?raw=true)


There is still something missing. One may notice that it is not particularly tailored to the changes in variance as the features change. Thus, in order to incorporate a more conditional error dependence, quantile regression may be considered. It is important to note that the guarantee provided by the multiple hypothesis method remains marginal over the features, even though the quantile regression provides some dependence visually. Here is the result for the perfect predictor followed by the imperfect predictor but with the quantile regression also being trained on the training data and then using the hypothesis testing wrapper on the held out set:
![image alt](https://github.com/Eminence-Spirit/Multiple-Hypothesis-Testing-Guarantee-for-Miscoverage-Loss-on-Arbitrary-Predictor-models./blob/main/images_for_READ_ME/Perf_qunatile.png?raw=true)
![image alt](https://github.com/Eminence-Spirit/Multiple-Hypothesis-Testing-Guarantee-for-Miscoverage-Loss-on-Arbitrary-Predictor-models./blob/51133650f27a6a9e571e0e51ae53ac31f36df340/images_for_READ_ME/Imperf_quantile.png)


The above images are for 1 dimensional feature and response, but on 2 dimensional feature and 1 dimensional response, [visualizations can be found here](https://drive.google.com/drive/u/1/folders/1uOAcTWUVdEJ2a-54snm3IBPzZghIl_5g)


# Hypothesis Setup
To obtain the desired guarantee on the population miscoverage loss, we formulate the problem as a multiple hypothesis testing problem.

To achieve this, suppose that we have a collection of candidate prediction sets
```math
C_1(\hat{f}(X)), C_2(\hat{f}(X)), \ldots, C_m(\hat{f}(X)).
```
For each candidate prediction set, let $\theta_i$ denote its population miscoverage probability:

```math
\theta_i(Z)
=
\mathbb{P}_{(X,Y)}\left(
Y \notin C_{Z,i}(\hat{f}(X)) \;\Bigg|\; Z
\right).
```

We can then formulate the hypothesis test for each candidate prediction set as

```math
H_{0,i}:\quad \theta_i > \epsilon
\qquad \text{vs.} \qquad
H_{a,i}:\quad \theta_i \leq \epsilon,
\qquad i=1,\ldots,m.
```
Each hypothesis represents the statement that the corresponding prediction set fails to achieve the desired population miscoverage level.

After applying the multiple hypothesis testing procedure, the selected prediction set is denoted by $C_Z(\hat{f}(X))$. The subscript $Z$ emphasizes that the selected set depends on the held-out data $Z$, while its construction also depends on the trained predictor $\hat{f}$. Thus, $C_Z(\hat{f}(X))$ is a subset of the response space $\mathbb{R}^k$. The selected prediction set is the one with the largest conditional miscoverage probability among the candidate sets whose null hypotheses are rejected. Since rejecting $H_{0,i}$ provides evidence that $\theta_i(Z) \leq \epsilon$, this corresponds to selecting the candidate with the greatest miscoverage probability among those satisfying the desired risk constraint.

Note that the estimator of the quantity is the sum of the indicator functions, and this results in using the binomial distribution, which has a UMP test for each hypothesis tested individually. Since the hypotheses are considered simultaneously, testing each one independently at level $\alpha$ does not generally provide the desired overall error control. However, since the hypotheses are ordered before even seeing the data, this method of testing the hypotheses from the one that has the lowest risk to the one that has the highest risk at level $\alpha$ can be shown to be better than any other hypothesis testing method within a class of graphical testing procedures known as sequential graphical testing procedures.

Thus, the problem of obtaining a high-probability guarantee on population miscoverage is transformed into a multiple hypothesis testing problem: by controlling the appropriate multiple-testing error rate, we can identify prediction sets for which the desired population miscoverage guarantee holds with high probability.

# Code
The full code is added in one file, it contains the testing datasets that were created, evaluation procedure, quantile regression and hypothesis testing for ease of use. Note that the code to run, you need to activate the tf environment 

### *Note on Conditionality
Every probabilistic statement made in this document is conditional on the training data, $T$ (and also the trained f_hat if the training procedure itself is probabilistic), though it is not explicitly written for ease of reading. This conditionality of the probabilistic claims of risk aligns with practical usage of pretrained machine learning models since claims need to be made on the specific model that is trained.
