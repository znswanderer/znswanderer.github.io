---
layout: post
title:  "Short Note regarding Commutators in Finite Dimensions"
date:   2021-08-01
categories: Physics
tags:   [Quantum Mechanics, Commutators, Discretization of Space]
---
{% raw %}
[Last time][last] we started the study of quantum mechanics in discretized space. In this blog post
I will take a very short look at what this means for non-commuting operators.

[last]: https://znswanderer.github.io/physics/Discrete/

{% endraw %}
<!--more-->
{% raw %}

If we express an operator in its discretized eigenbasis, like the position operator
in the discrete position space, this operator will be *diagonal*:

$$
X = \Delta x
\begin{pmatrix} 
0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 & 0 \\
0 & 0 & 0 & 3 & 0 \\
0 & 0 & 0 & 0 & 4  
\end{pmatrix}
+ x_0 \cdot \textbf{1} 
$$

where $$x_0$$ is just the origin of our box of size $$L$$ and $$\Delta x = L / N$$ with $$N$$ the size
of the discretization (in the example above we have $$N=5$$).

The canonical conjugate momentum is not diagonal in the position basis. In the last post
we have seen two versions of the momentum operator:


$$
P_+ =
- \frac{i \hbar}{\Delta x}
\begin{pmatrix}
-1 & 1 & 0 & 0 &  0 \\
0 & -1 & 1 & 0 &  0 \\
0 & 0 & -1 & 1 &  0 \\
0 & 0 & 0 & -1 & 1  \\
0 & 0 & 0 & 0 & -1  
\end{pmatrix}
$$

for the forward difference and one for the backward difference:

$$
P_- = 
- \frac{i \hbar}{\Delta x}
\begin{pmatrix}
1 & 0 & 0 & 0 &  0 \\
-1 & 1 & 0 & 0 &  0 \\
0 & -1 & 1 & 0 &  0 \\
0 & 0 & -1 & 1 & 0  \\
0 & 0 & 0 & -1 & 1  
\end{pmatrix}
$$

Clearly neither of them is purely diagonal.





This problem is at the very heart of of quantum mechanics. Two non-commuting 
operators do not share a common eigenbasis and therefore they can not be
diagonal at the same time. (A proof
for the inverse of this argument, that if operators $$\hat A$$ and $$\hat B$$ share 
a basis it follows that $$[\hat A, \hat B] = 0$$, can be found on [wikipedia][commuting].
The proof, that if $$[\hat A, \hat B] \neq 0$$ both cannot be diagonal for a common
basis, can be made along the same lines.)

[commuting]: https://en.wikipedia.org/wiki/Complete_set_of_commuting_observables#Proofs

The fact, that these two operators can not be diagonal at the same time implies, that
these operators do not commute. But if we now take a look at the commutator
in a discrete space, we find something interesting.

The canonical commutation relation for the momentum and position operator is

$$
[\hat x, \hat p] = i \hbar \, .
$$

So if we calculate the commutator for the matrix representation we would expect something like

$$
[X, P_\pm] = i \hbar \textbf{1}
$$

where $$\textbf{1}$$ is the unit matrix.

Let's check this using SymPy!


```python
from sympy import symbols, I, simplify
from sympy.matrices import Matrix, eye

dx, x0 = symbols(r'\Delta{x} x_0')
N = 5
hbar = 1

def x_op(n, m):
    # this is the kronecker delta
    if n == m:
        return m
    else:
        return 0
    
def p_fwd(n, m):
    if n == m:
        return -1
    elif n+1 == m:
        return 1
    else:
        return 0

def p_bwd(n, m):
    if n == m:
        return -1
    elif n-1 == m:
        return 1
    else:
        return 0

P_f = -I * hbar * Matrix(N, N, p_fwd) / dx
P_b = -I * hbar * Matrix(N, N, p_bwd) / dx
X = dx * Matrix(N, N, x_op) + x0 * eye(N)
```

We can now examine the commutators, first $$[X, P_+]$$:


```python
simplify(X @ P_f - P_f @ X)
```
$$\displaystyle \left[\begin{matrix}0 & i & 0 & 0 & 0\\0 & 0 & i & 0 & 0\\0 & 0 & 0 & i & 0\\0 & 0 & 0 & 0 & i\\0 & 0 & 0 & 0 & 0\end{matrix}\right]$$  



and then $$[X, P_-]$$:


```python
simplify(X @ P_b - P_b @ X)
```
$$\displaystyle \left[\begin{matrix}0 & 0 & 0 & 0 & 0\\- i & 0 & 0 & 0 & 0\\0 & - i & 0 & 0 & 0\\0 & 0 & - i & 0 & 0\\0 & 0 & 0 & - i & 0\end{matrix}\right]$$  



As one can see, neither of the commutators is equal to $$i\hbar \textbf{1}$$. The diagonal is empty for both commutators. 
This means, that the trace of both commutators is $$0$$. 

This is not a consequence of
the specific form of the momentum operators we have chosen, but is valid for all
matrices in finite dimensional spaces, because for
every two matrices $$A$$ and $$B$$ the following holds:

$$
\textrm{Tr}(AB) = \textrm{Tr}(BA)
$$

and with $$\textrm{Tr}(C + D) = \textrm{Tr}\,C + \textrm{Tr}\,D$$ it follows

$$
\textrm{Tr}[A,B] = 0
$$

This is rather interesting. Here Weinberg
notes:

> The trace of the unit operator $$\textbf{1}$$ is just $$\sum_i 1$$, which is the
> dimensionality of the Hilbert space, and hence is not defined in Hilbert
> spaces of infinite dimensionality. Note in particular that in a space of
> finite dimensionality the trace of the commutation relation $$[X, P] = i \hbar \textbf{1}$$
> would give the contradictory result $$0 = i \hbar \textrm{Tr}\, \textbf{1}$$, so this
> commutation relation can only be realized in Hilbert spaces of infinite dimensionality, 
> where the traces do not exist.  
> Steven Weinberg, *Lectures on Quantum Mechanics*, 1st edition, § 3.3 Observables, pg. 67

So the "correct" commutation relation $$[\hat x, \hat p] = i \hbar$$ can only be realized in
the continuum.
But the non-commuting nature of the operators is still there in an approximation, 
because the non-zero diagonals in $$[X, P_\pm]$$ are direct neighbours of the main diagonal.
For large $$N$$ the difference from the main diagonal can be made as small as needed.

*The original Jupyter notebook can be found [here](<https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter/2021-08-01-Regarding-Commutators.ipynb>).*
 {% endraw %}