---
layout: post
title:  "The Momentum Operator as Generator of Space Translation"
date:   2022-08-23
categories: Physics
tags:   [Quantum Mechanics]
---
{% raw %}
We take a look at an operation, which translates a state, represented by a wave function $$\psi(x)$$ in position space,
by an amount $$\Delta x$$. The new state is then represented by the wave function $$\tilde \psi(x)$$.

{: style="text-align:center"}
{% endraw %}
![Image]({{site.url}}/assets/images/2022-08-23-Momentum-Operator_files/psi_trans_aug2022.png)
{% raw %}

{% endraw %}
<!--more-->
{% raw %}

This operation shall be described by the operator $$\hat U_{\Delta x}$$:

$$
\tilde \psi(x) = \hat U_{\Delta x} \psi (x) \, .
$$

We can relate both functions by noting, that

$$
\tilde \psi(x) = \psi (x - \Delta x) \, .
$$

For small $$\Delta x$$ we can construct the shifted function by using derivative

$$
\psi (x - \Delta x) \approx \psi (x) - \Delta x \frac{\partial}{\partial x} \psi(x)
$$

The momentum operator in position space is given by

$$
\hat p = -i \hbar \frac{\partial}{\partial x}
$$

Therefore we can write

$$
\tilde \psi (x) \approx \left( 1 -  i \Delta x \hat p  / \hbar \right) \psi(x)
$$

If we split the translation $$\Delta x$$ into $$N$$ small successive translations, each of amount $$\Delta x/N$$

$$
\left( 1 - i \frac{\Delta x}{N} \hat p / \hbar\right) \left( 1 - i \frac{\Delta x}{N} \hat p / \hbar\right)
\dots \left( 1 - i \frac{\Delta x}{N} \hat p / \hbar\right) = \left( 1 - i \frac{\Delta x}{N} \hat p / \hbar\right)^{N}
$$

and take the limit $$N \to \infty$$, we can identify that with the [exponential function](https://en.wikipedia.org/wiki/Characterizations_of_the_exponential_function):

$$
\lim_{N \to \infty} \left( 1 - i \frac{\Delta x}{N} \hat p / \hbar\right)^{N}
= \exp \left(- i \Delta x \hat p / \hbar \right)
$$

So we find

$$
\boxed{\hat U_{\Delta x} = \exp \left(- i \Delta x \hat p / \hbar \right)} \, .
$$

## Operator as a function argument

### Taylor series

Having an operator as an argument for the exponential function might seem strange, but there are two ways 
to look at this. First we know that the expansion of the exponential function is

$$
\exp x = 1 + x + \frac{1}{2!} x^2 + \frac{1}{3!} x^3 + \dots
$$

The substition of the momentum operator with the derivative (in position space) in the exponential then gives

$$
\exp \left(- i \Delta x \hat p / \hbar \right) | \psi \rangle
\longrightarrow 
\exp \left( - \Delta x \frac{\partial}{\partial x} \right) \psi (x)
$$

and this is just the [Taylor series](https://en.wikipedia.org/wiki/Taylor_series)
expansion for $$\psi (x - \Delta x)$$

$$
\psi (x - \Delta x) = \psi(x) - \Delta x \frac{\partial}{\partial x} \psi(x) + 
\frac{{\Delta x}^2}{2!} \frac{\partial^2}{\partial x^2} \psi(x) + \dots
$$

### Momentum Space

The strangeness of having operators inside the exponential vanishes, if we move to momentum space. Using 
the basis vectors $$\hat p \vert p \rangle = p \vert p \rangle$$ with $$\langle p' \vert p \rangle = \delta_{p',p}$$ we write

$$
\begin{align}
\exp \left(- i \Delta x \hat p / \hbar \right) | \psi \rangle 
&= \sum_{p', p} |p'\rangle \langle p'| \exp \left(- i \Delta x \hat p / \hbar \right) |p\rangle \langle p| \psi \rangle \\
&= \sum_{p} \exp \left(- i \Delta x p / \hbar \right) \psi(p) |p\rangle
\end{align}
$$

This can be done, because we the operator $$\hat U_{\Delta x} = \exp \left(- i \Delta x \hat p / \hbar \right)$$
commutes with $$\hat p$$ and therefore the eigenvectors of $$\hat p$$ are also eigenvectors of $$\hat U_{\Delta x}$$.
In the last line, the arguments of the exponentials contain ordinary numbers $$p$$ as the eigenvalues of $$\hat p$$.


## Unitary Operators

If we can shift the state by $$\Delta x$$ to the right, we can also reverse the operation by applying a shift of
$$-\Delta x$$, which will return the original state. The reversal of an operation $$\hat U$$ is expressed by $$\hat U^{-1}$$.
Here we have

$$
\hat U_{\Delta x}^{-1} = \hat U_{-\Delta x}
$$

and

$$
\hat U_{\Delta x}^{-1} \hat U_{\Delta x} \psi(x) = \psi (x) \, .
$$

If we take a a look at the expression of $$\hat U_{\Delta x}$$ in terms of the momentum operator we find

$$
\hat U_{\Delta x}^{-1} = \hat U_{-\Delta x} = \exp \left(i \Delta x \hat p / \hbar \right)
$$

This means, as $$\hat p^\dagger = \hat p$$ is a Hermitian operator we find

$$
\hat U_{\Delta x}^{-1} = \hat U_{\Delta x}^\dagger \, .
$$

An operator with $$\hat A^{-1} = \hat A^\dagger$$ is called [*unitary*](https://en.wikipedia.org/wiki/Unitary_operator). 
All eigenvalues of an unitary operator are purely imaginary, if $$\lambda_p$$ is 
an eigenvalue of the momentum operator, $$\hat p \vert p \rangle = \lambda_p \vert p \rangle$$, we have 


$$
\hat U_{\Delta x} \vert p \rangle = \exp \left(i \Delta x \lambda_p / \hbar \right) \vert p \rangle
$$

and as $$\lambda_p \in \mathbb{R}$$, because $$\hat p$$ is Hermitian, the eigenvalue $$\exp \left(i \Delta x \lambda_p / \hbar \right)$$
is purely imaginary.

The operators $$\hat U_{\Delta x}$$ form a [Lie group](https://en.wikipedia.org/wiki/Symmetry_in_quantum_mechanics#Overview_of_Lie_group_theory).
Lie groups are continuous and a group parameter $$\zeta \in \mathbb{R}$$ defines the individual group element $$G(\zeta)$$. For the parameter $$\zeta = 0$$ the group element is the identitiy element, $$G(0) = 1$$.
The *generator* $$X$$ of a Lie group describes the linear approximation for the group elements on the parameter at the identity element:

$$
X = \left. \frac{\partial G}{\partial \zeta} \right|_{\zeta = 0}
$$

Here the Lie group is the unitary translation operator $$\hat U_{\Delta x}$$ and the group parameter is $$\Delta x$$. Taking the derivative
we find the generator to be $$-i \hat p / \hbar$$. So the generator of spatial translations is the momentum operator (with some conversion factors).

# Code example

We want to shift a wave function $$\psi(x) = \langle x \vert \psi \rangle$$ using the momentum space. The shifted wave function is

$$
\tilde \psi(x) = \langle x | \hat U | \psi \rangle
$$

Using the unit operator in $$p$$ and $$x$$-space, we find

$$
\int_p dp \langle x | p \rangle  \int_{p'} dp' \langle p | \hat U | p' \rangle \int_{x'} dx'  \langle p' | x' \rangle \langle x' | \psi \rangle
$$

the $$\vert p \rangle$$ are eigenvectors of $$\hat U$$ and $$\langle p \vert p' \rangle = \delta(p - p')$$, so this simplifies to

$$
\int_p dp \langle x | p \rangle \langle p | \hat U | p \rangle \int_{x'} dx'  \langle p | x' \rangle \langle x' | \psi \rangle
$$


## Discrete Space

[Previously](https://znswanderer.github.io/physics/Discrete/) we have discussed
how to represent the wavefunction $$\psi(x)$$ in a discretized space with $$N$$ bins:

$$
\langle x | \psi \rangle = \psi(x) \mapsto 
\mathbf{\psi} = \begin{pmatrix}
\psi_0 \\
\psi_1 \\
\vdots \\
\psi_{N-1}
\end{pmatrix}
$$

The $$l$$-th complex conjugated momentum eigenfunction $$\langle p_l \vert x \rangle$$ can then be
described by a row vector and the scalar procuct becomes the multiplication of
a row with a column vector:

$$
\int_{x'} dx'  \langle p_l | x' \rangle \langle x' | \psi \rangle \mapsto 
\begin{pmatrix}
p_{l, 0}^* &
p_{l, 1}^* &
\dots &
p_{l, N-1}^*
\end{pmatrix}
\begin{pmatrix}
\psi_0 \\
\psi_1 \\
\vdots \\
\psi_{N-1}
\end{pmatrix}
$$

In discrete space of binning $$N$$, the momentum operator also has $$N$$ eigenvectors and we can stack
the $$N$$ eigenvectors to form a quadratic matrix

$$
\mathbf{Q}^{-1} 
= \begin{pmatrix}
p_{0, 0}^* &
p_{0, 1}^* &
\dots &
p_{0, N-1}^* 
\\
p_{1, 0}^* &
p_{1, 1}^* &
\dots &
p_{1, N-1}^* 
\\
\vdots \\
p_{N-1, 0}^* &
p_{N-1, 1}^* &
\dots &
p_{N-1, N-1}^* 
\end{pmatrix}
$$

with this we can write the complete expression for $$\mathbf{\tilde \psi}$$ in discrete space as

$$
\boxed{\mathbf{\tilde \psi} = \mathbf{Q} \mathbf{\Lambda_U} \mathbf{Q}^{-1} \mathbf{\psi}} \, ,
$$

where $$\mathbf{\Lambda_U}$$ is the diagonal matrix $$\langle p \vert \hat U \vert p \rangle$$.
This is known as [Eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Eigendecomposition_of_a_matrix).

The momentum eigenfunction are

$$
-i \hbar \frac{\partial}{\partial x} e^{ikx} = \hbar k e^{ikx} \, .
$$

If we use periodic boundary conditions, $$\psi(x) = \psi(x + L)$$, the values for $$k$$ a discrete: $$k = \pm 2 \pi n / L$$.
This disceteness is not because of the discretization of space, but because of the periodicity. The
values for $$n$$ are not bounded, but now because of the discretization of space, there are only $$N$$ linear indepentent 
eigenvectors. We will choose the eigenvectors for $$n= -\frac{N}{2}, -\frac{N}{2} +1, \dots , \frac{N}{2} -1$$. 


```python
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

N = 300
L = 1
hbar = 1
x = np.linspace(0, L, N, endpoint=False)
dx = np.diff(x)[0]

ns = range(-N//2, N//2)
ks = [(2 * np.pi / L) * n for n in ns]
psis = [np.exp(1j * k * x).T.conj() / np.sqrt(N) for k in ks]
Q_inv = np.array(psis)
Q = Q_inv.T.conj()

delta_x = 0.1
U_dx = sparse.diags(np.exp(-1j * delta_x * np.array(ks)))
```

The test function shall be a simple gaussian. In this function we can now apply the shift operator for $$\Delta x = 0.1$$:


```python
gauss = np.exp(-(x-0.5)**2/0.01)
plt.figure(figsize=(7,5))
plt.plot(x, gauss, label=r"$\psi(x)$")
gauss2 = Q @ U_dx @ Q_inv @ gauss
plt.plot(x, gauss2.real, label=r"$\tilde \psi(x)$")
plt.legend();
```


    
{: style="text-align:center"}
{% endraw %}
![Image]({{site.url}}/assets/images/2022-08-23-Momentum-Operator_files/2022-08-23-Momentum-Operator_20_0.png)
{% raw %}
    


And this is the result, that we wanted to achieve.

*The original Jupyter notebook can be found [here](<https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter/2022-08-23-Momentum-Operator.ipynb>).*
 {% endraw %}