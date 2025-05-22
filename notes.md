# A Quick Walk through for Using Fisher Information for Active Perception

## Preliminaries on Definitions for Information

We have the following basic definitions:

Shannon information:
$$
H[x] := -\log p(x)
$$

Conditional entropy:
$$
H[X|Y] = \mathbb{E}_{p(x, y)}[-\log p(x|y)]
$$

Information Gain:
$$
\mathbb{I}[X; Y] := H[X] - H[X|Y]
$$

Conditional Information Gain:
$$
\mathbb{I}[X; Y|Z] := H[X|Z] - H[X|Y, Z]
$$

Conditional entropy with Bayes' Theorem:
$$
H[A|B] = H[B|A] - H[B] + H[A]
$$

Conditional entropy with multiple variables:
$$
H[A|B, C] = H[B|A, C] - H[B|C] + H[A|C]
$$

## Information Gain

We can simplify the notation a bit and use $x_c$ and $y_c$ to denote the candidate data point and its label, where we typically have access to $x_c$ but not $y_c$. 
Meanwhile, we denote parameters as $w$ and the existing training dataset as $D$, which is a collection of pairs $(x_i, y_i)$.
An example of $x_c$ and $y_c$ can be the $SE(3)$ camera pose and the RGB image, respectively.

When selecting a candidate data point $x_c$ for an active perception task, we can maximize the information gain by:

$$
\arg \max_{x_c} \mathbb{I}[w; y_c| x_c, D]
$$

The information gain can be expressed as:

$$
\begin{equation}
    \mathbb{I}[w; y_c| x_c, D] = H[w| x_c, D] - H[w| y_c, x_c, D] 
\end{equation}
$$

Assume that $x_c$ is chosen independently from $D$:

$$
H[w|x_c, D] = H[w | D]
$$

Thus we have:

$$
\begin{equation}
    \mathbb{I}[w; y_c| x_c, D] = H[w| D] - H[w| y_c, x_c, D] 
\end{equation}
$$

Thus the problem is how we can approximate and compute the entropy $H[w| D]$ for any dataset $D$.

### Approximation of $H[w|D]$

We approximate the distribution of $p(w|D)$ with a multivariate Gaussian distribution.

Taking Taylor expansion of $H[w|D]$ around the current estimate $w^*$, we have:

$$
H[w|D] \approx H[w|D] + H'[w|D] (w - w^*) + \frac{1}{2} (w - w^*)^T \mathbf{H}''[w^*|D] (w - w^*)
$$

where $\mathbf{H}''[w^*|D]$ is the Hessian matrix of $H[w|D]$ at $w^*$. 

The first term is a constant given the current estimate $w^*$, and the second term is close to zero when our model is at a local minimum because we get $w$ by optimizing the model with $D$.

Therefore, we can approximate the distribution of $p(w|D)$ with a multivariate Gaussian distribution with mean $w^*$ and covariance matrix $\mathbf{H}''[w^*|D]^{-1}$.

$$
\begin{equation}
    p(w|D) \sim \mathcal{N}(w^*, \mathbf{H}''[w^*|D]^{-1})
\end{equation}
$$

Specifically, the $p(w | D)$ is given as:

$$
p(w|D) = \frac{1}{\sqrt{(2\pi)^n |\mathbf{H}''[w^*|D]|}} \exp\left[-\frac{1}{2} (w - w^*)^T \mathbf{H}''[w^*|D] (w - w^*)\right]
$$

where $n$ is the number of parameters in $w$.

The entropy of a multivariate Gaussian distribution is known to be:

$$
\begin{align}
    H[w|D] &= \mathbb{E}_{p(w|D)}[-\log p(w|D)] \\
    & \approx \mathbb{E}_{p(w|D)} \left[-\log \frac{1}{\sqrt{(2\pi)^n |\mathbf{H}''[w^*|D]|}} \exp  \left( - \frac{1}{2} (w - w^*)^T \mathbf{H}''[w^*|D] (w - w^*) \right) \right] \\
    &=  -\frac{1}{2} \log \det |\mathbf{H}''[w|D]| + \frac{n}{2} \log(2\pi) + \frac{1}{2} n \\
    &= -\frac{1}{2} \log |\mathbf{H}''[w|D]| + C_n 
\end{align}
$$

where $C_n$ is a constant that depends on the number of parameters $w$.

### Approximation of $\mathbf{H}''[w|D]$ from Empirical Fisher Information

We can rewrite the entropy $H[w|D]$ with Bayes' theorem:

$$
\begin{equation}
    H[w|D] = H[D|w] + H[w] - H[D]
\end{equation}
$$

$D$ is a pre-determined dataset thus $H[D]$ is constant. We have no prior knowledge of the distribution of $w$ thus we can assume $H[w]$ with an uninformative prior such as a uniform distribution.

Taking the second derivative of the above equation with respect to $w$, we have:
$$
\mathbf{H}''[w|D] \approx \mathbf{H}''[D|w] + \mathbf{H}''[w]
$$

where $\mathbf{H}''[w] = 0$ for the uninformative prior. Therefore we can approximate the Hessian matrix $\mathbf{H}''[w|D]$ with $\mathbf{H}''[D|w]$. 
$$
\begin{equation}
    \mathbf{H}''[w|D] \approx \mathbf{H}''[D|w]
\end{equation}
$$

where $\mathbf{H}''[D|w]$ can be computed by approximating the distribution of $p(D|w)$. 

Again, $D$ is a collection of pairs $(x_i, y_i)$, where $x_i$ is the input and $y_i$ is the label. 
Without loss of generality, we can have any pair $(x_i, y_i)$ and denote it as $(x, y)$ and the joint probability:

$$
p(y, x|w) = p(y|x, w) p(x|w)
$$

as the rendering model does not model the distribution of $x$ and we assume each $x_i$ is uniformly sampled from the dataset $D$.

$$
p(y, x|w) = p(y|x, w) p_{\text{data}}(x)
$$

where $p_{\text{data}}(x)$ is the distribution of $x$ in the dataset $D$. Thus we can compute $\mathbf{H}''[D|w]$ by computing the Hessian matrix of our negative log-likelihood function. In our case it is the mean squared error (MSE) thus the Hessian is not dependent on the label $y$.

### Approximation of $H[w|y_c, x_c, D]$

This is similar to the approximation of $H[w|D]$, except we do not have access to $y_c$ because that is the ground truth label we have not observed yet. However, later we can see our computation does not depend on $y_c$.

Similarly, we can approximate the distribution of $p(w|y_c, x_c, D)$ with a multivariate Gaussian distribution. Thus, we only need to find out the Hessian matrix $\mathbf{H}''[w|y_c, x_c, D]$.

Rewrite the entropy with Bayes' theorem:

$$
\begin{equation}
   H[w| y_c, x_c, D] = H[y_c| x_c, w, D] - H[y_c|x_c, D] + H[w|x_c, D] 
\end{equation}
$$

We then look into each term in the above equation. 
First, we assume that all the information of $D$ has been observed by $w$ which is what we did in practice, where we only use the model $w$ instead of $D$ after training.
Second, we find $H[y_c|x_c, D]$ is not dependent on $w$ because it is the ground truth label we have not observed yet. 
Third, $x_c$ has no contribution to $w$ because the model is not trained with $x_c$ without the label $y_c$. $H[w|x, D] = H[w|D]$.

Therefore, we have:

$$
\begin{equation}
    H[y_c|x_c, w, D] = H[y_c|x_c, w] - H[y_c|x_c, D]  + H[w| D]
\end{equation}
$$

By taking the derivative of the above equation with respect to $w$ twice, we have:

$$
\begin{equation}
    \mathbf{H}''[y|x_c, w, D] = \mathbf{H}''[y_c|x_c, w] + \mathbf{H}''[w| D]
\end{equation}
$$

Both of the terms can be computed by the Hessian matrix of the negative log-likelihood function.

### Computing the Expected Information Gain

Bringing the above equations together, we have:

$$
\begin{align}
    & \mathbb{I}[w; y_c| x_c, D] = H[w| x_c, D] - H[w| y_c, x_c, D] \\
    &= H[w| D] -H[w|y_c, x_c, D] \\
    &\approx \left(-\frac{1}{2} \log \det \mathbf{H}''[w| D] + C_n \right) - \left(-\frac{1}{2} \log \det \mathbf{H}''[w| y_c, x_c, D] + C_n \right) \\
    &= -\frac{1}{2} \log\det \mathbf{H}''[w| D] + \frac{1}{2} \log \det \left( \mathbf{H}''[w| y_c, x_c] + \mathbf{H}''[w| D] \right) \\
    &= -\frac{1}{2} \log\det \left(\mathbf{H}''[D|w]^{-1} \mathbf{H}''[y_c| x_c, w] + I\right)
\end{align}
$$

where $I$ is the identity matrix. The expectation is taken when we consider the Gaussian distribution of $p(w|D) \sim \mathcal{N}(w^*, \mathbf{H}''[w|D]^{-1})$.

For ease of computation, we can further use the trace instead of the log determinant:
$$
\begin{equation}
    -\frac{1}{2} \log\det \left(\mathbf{H}''[D|w]^{-1} \mathbf{H}''[y_c| x_c, w] + I\right) \leq -\frac{1}{2} \text{tr}\left(\mathbf{H}''[D|w]^{-1} \mathbf{H}''[y_c| x_c, w]\right)
\end{equation}
$$

One quick proof is for any square matrix $A$ we have:

$$
\begin{align}
    \log \det (A+I) \leq \log \prod_{i} (\lambda_i + 1) = \sum_{i} \log (\lambda_i + 1) \leq \sum_{i} \lambda_i = \text{tr}(A)
\end{align}
$$

where the equality holds when $A = 0$.

### Approximate the Hessian Matrix $\mathbf{H}''[y|x, w]$

Assume our model is a function $\tilde{y} = f(x; w)$, where $x$ is the input and $w$ is the parameter. We can compute the Hessian matrix $\mathbf{H}''[y|x, w]$ for the negative log-likelihood $-\log p(y|x, w)$ as:

$$
\begin{align}
    \mathbf{H}''[y|x, w] &= \nabla^2_{w} -\log p(y|x, w) \\
    &= \nabla_{w} f(x; w)^T \; \mathbb{E}_{p\left(y|f(x;w)\right)} \left[ \nabla_{f(x;w)}^2 p(y | f(x;w))\right] \; \nabla_{w} f(x; w) \\
\end{align}
$$

where $\nabla^2_{w}$ is the second derivative with respect to $w$. $\mathbb{E}_{p\left(y|f(x;w)\right)} \left[ \nabla_{f(x;w)}^2 p(y | f(x;w))\right] = I$ because our negative log-likelihood is MSE.
Thus we have:

$$
\begin{equation}
    \mathbf{H}''[y|x, w] = \nabla_{w} f(x; w)^T \; \nabla_{w} f(x; w)
\end{equation}
$$

where $\nabla_{w} f(x; w)$ is the Jacobian matrix of $f(x; w)$ with respect to $w$. 

$\mathbf{H}''[D | w]$ can be computed similarly as we assume each data point $(x_i, y_i)$ is independent. Another interpretation is Fisher Information is additive. 

# Summary

We can compute the expected information gain by:

$$
\begin{equation}
    \mathbb{I}[w; y_c| x_c, D] \approx -\frac{1}{2} \text{tr}\left(\mathbf{H}''[D|w]^{-1} \mathbf{H}''[y_c| x_c, w]\right)
\end{equation}
$$

where $\mathbf{H}''[y_c| x_c, w] = \nabla_{w} f(x_c; w) \; \nabla_{w} f(x_c; w)^T$ and $\mathbf{H}''[D|w] = \sum_{i} \nabla_{w} f(x_i; w)^T \; \nabla_{w} f(x_i; w)$.
