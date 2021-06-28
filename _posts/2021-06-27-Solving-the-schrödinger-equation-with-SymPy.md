---
layout: post
title:  "Solving the Schrödinger Equation with Sympy"
date:   2021-06-27
categories: blog
tags:   [physics, ODE, SymPy, Quantum Mechanics]
---


Today I will briefly show how to use [SymPy][sympy] for solving the Schrödinger equation.

Basically there are two ways to approach the Schrödinger equation with a computer: numerically and analytically/symbolically. For many problems often only the purely numerical way remains, because the corresponding equations cannot be solved analytically or only by analytical approximation methods.

<!--more-->

But to get a feeling for the symbolic solution of differential equations like the Schrödinger equation, it is advisable to start with the simplest case of a Schrödinger equation: the 
[quantum free particle][free_particle].

[sympy]: https://www.sympy.org/en/index.html
[free_particle]: https://en.wikipedia.org/wiki/Free_particle#Quantum_free_particle

## The quantum free particle

A quantum particle in one spatial dimension is described by a wave function in position space:

$$
\psi(x, t)
$$

The Hamiltonian for the *free particle* is

$$
\hat{H} = \frac{1}{2m} \hat{p}^2
$$

With $$\hat p = -i \hbar \frac{\partial}{\partial x}$$ in position space, we get for the hamiltonian:

$$
\hat{H} = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2}
$$

the particle is called "free", because there is no potential energy term in the
Hamiltonian and the particle is *free* to move (in one dimension).

The [Schrödinger Equation][schroedinger] is

$$
    i \hbar \frac{\partial}{\partial t} \psi(x, t) = \hat{H} \psi(x,t)
$$

We make the *ansatz* of separating time and space dependency in $$\psi$$:

$$
\psi(x, t) = \psi(t) \, \psi(x)
$$

this way, because $$\hat{H}$$ does not effect time, we can first search for
solutions of the time-independent Schrödinger Equation:

$$
E \psi(x) = \hat{H} \psi(x) = -\frac{\hbar^2}{2m} \frac{\partial^2 \psi(x)}{\partial x^2}
$$

The solutions are called *eigenfunctions* of the Hamilton operator and the
energy values $$E$$ are the corresponding *eigenvalues*.

For these eigenfunctions the overall solution to the time-dependent Schrödinger equation is:

$$
\psi(x, t) = e^{-i E t / \hbar} \psi(x)
$$

So, once the solutions to the time-independent Schrödinger equation are found, the time-dependency
comes in rather easily. Finding the eigenfunctions of the Hamiltonian is often the most
important task.

[schroedinger]: https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation

### Solving in SymPy

Let's try using SymPy for analytical solutions to this eigenvalue problem!

First, import sympy and define functions, variables, parameters, constants...


```python
from IPython.display import display
import sympy as smp
from sympy.physics.quantum.constants import hbar

psi = smp.symbols(r'\psi', cls=smp.Function, complex=True)
x = smp.symbols('x', real=True)
E, m, L = smp.symbols('E m L', real=True, positive=True)
```

Please note, that we tell SymPy, that $$E$$, $$m$$ and $$L$$ are real and positive. The
position $$x$$ on the other hand can be negative, too. And the wavefunction $$\psi$$ is
complex and its class is `Function`. 

Now, let's define the Hamilton operator. 

For this we need the second derivative of the function $$\psi(x)$$ with respect to $$x$$.
This is done with the SymPy expression `diff(psi(x), x, x)`.


```python
H_psi = - (hbar**2/ (2 * m)) * smp.diff(psi(x), x, x)
H_psi
```

$$- \frac{\hbar^{2} \frac{d^{2}}{d x^{2}} \psi{\left(x \right)}}{2 m}$$



Finally, the Schrödinger equation for the free particle is defined as ... an equation! Who would have guessed?

But how do we express the equation

$$
E \psi = \hat H \psi
$$

in SymPy?

The naive way does not work:


```python
E * psi(x) = H_psi
```


      File "<ipython-input-3-1668d17d04a1>", line 1
        E * psi(x) = H_psi
        ^
    SyntaxError: cannot assign to operator
    


This is because the equal sign `=` in Python is really $$:=$$, an *assignment*.

Another possibility would be:


```python
E * psi(x) == H_psi
```




    False



This time we do not get an error message, but it is still not, what we were looking for: 
the double equal sign `==` is the *comparison* operator. Clearly the right-hand-side (RHS) 
is in general not equal to the LHS. In fact we are looking for the solutions $$\psi$$ for which the
equality will hold.

The correct way in SymPy is to define an `Eq` object, like `Eq(RHS, LHS)`. Let's do just that:


```python
eq_schroed = smp.Eq(E * psi(x), H_psi)
eq_schroed
```




$$E \psi{\left(x \right)} = - \frac{\hbar^{2} \frac{d^{2}}{d x^{2}} \psi{\left(x \right)}}{2 m}$$



Now, we want to use `SymPy.dsolve` to get the solution for this ordinary differential equation (ODE).


```python
sol = smp.dsolve(eq_schroed, psi(x))
sol
```




$$\psi{\left(x \right)} = C_{1} \sin{\left(\frac{\sqrt{2} \sqrt{E} \sqrt{m} x}{\hbar} \right)} + C_{2} \cos{\left(\frac{\sqrt{2} \sqrt{E} \sqrt{m} x}{\hbar} \right)}$$



This alreads looks kind of right, but the factors in the arguments are a bit awkward. 
We can replace this with the wavenumber:

$$
k = \frac{\sqrt{2 m E}}{\hbar}
$$

and substitute this in the solution:


```python
# the k defined above must be positive, because E is real
k = smp.symbols('k', real=True, positive=True)  
sol = sol.subs(smp.sqrt(2 * E * m) / hbar, k)
sol
```




$$\psi{\left(x \right)} = C_{1} \sin{\left(k x \right)} + C_{2} \cos{\left(k x \right)}$$



The usual way to decribe the position part of a quantum free particle (in one dimension) is:

$$
\psi(x) = A e^{i k x}
$$

To see that both solutions are equivalent, one has to remember, that there is also a solution for $$-k$$ having the same energy.
So we can always find solutions for the constants that will give:

$$
C_1 \sin(kx) + C_2 \cos(k x) = A_1 e^{ikx} + A_2 e^{-ikx}
$$

(The $$C$$´s and $$A$$´s can be complex, too) 


## Particle in a box

Now we demand, that the particle is confined to a [box][box], this means we have the constraint:

$$
\psi(0) =0 \quad \text{and} \quad \psi(L) = 0
$$

The original solution for the free particle is stored in `sol.rhs`

[box]: https://en.wikipedia.org/wiki/Particle_in_a_box


```python
sol.rhs
```




$$C_{1} \sin{\left(k x \right)} + C_{2} \cos{\left(k x \right)}$$



We are now using SymPy´s `solve` function to find the values for $$C_1$$, $$C_2$$ and $$k$$ that will 
satisfy the constraints given by the box for the free particle.

These constraints are combined in a list, which means an *AND* combination: both 
constraints must hold at the same time.


```python
constraints = [smp.Eq(sol.rhs.subs(x, 0), 0), smp.Eq(sol.rhs.subs(x, L), 0)]
for c in constraints: display(c)
```


$$C_{2} = 0$$



$$C_{1} \sin{\left(L k \right)} + C_{2} \cos{\left(L k \right)} = 0$$


(Here we could also use `constraints = [sol.rhs.subs(x, 0), sol.rhs.subs(x, L)]` as the constraints as just giving the term means implicitly equality with 0)

Now we use these constraints in SymPy´s solve function to find the values for all parameters:


```python
C1, C2 = smp.symbols('C1 C2')
smp.solve(constraints, (C1, C2, k))
```




    [(0, 0, k), (C1, 0, pi/L)]



We get two solutions. The first one is trivial and not too useful, as it means $$\psi(x) = 0$$

The second solution is: $$C_2 = 0$$, $$k = \pi / L$$, which gives $$\psi(x) = C_1 \sin(\pi x / L)$$. This is surely correct, but only one possible solution.

After some ~google searching~ research, I found SymPy´s [solveset][solveset]. Under the heading *What’s wrong with solve()*
there is explicitly stated, that `solve` has issues with
 
> Infinitely many solutions: $$\sin(x)=0$$  

So maybe `solveset` will be able to find *all* solutions for $$k$$.

As it turns out, `solveset` itself can handle only one constraint, but we have two constraints. 
Luckily, same module has `nonlinsolve` which works perfectly:

[solveset]: https://docs.sympy.org/latest/modules/solvers/solveset.html


```python
from sympy.solvers.solveset import nonlinsolve
nonlinsolve(constraints, (C1, C2, k))
```




$$\left\{\left( 0, \  0, \  k\right), \left( C_{1}, \  0, \  \left\{\frac{2 n \pi + \pi}{L}\; |\; n \in \mathbb{Z}\right\}\right), \left( C_{1}, \  0, \  \left\{\frac{2 n \pi}{L}\; |\; n \in \mathbb{Z}\right\}\right)\right\}$$



We can combine the last two solutions to

$$
\psi(x) = C_1 \sin \left(\frac{n \pi}{L} x \right)
$$

which is, of course, the correct solution for the particle in a box.


## Conclusion

SymPy is a very powerful python package and it is really fantastic, that we get all this for free! 

On the other hand, even solving a rather simple problem took me quite some time and I feel like 
getting good in using SymPy would mean some substantial effort.

The Jupyter notebook for this post can be found [here][notebook].

## References:

* Introduction on how to use ODE solvers in SymPy: <http://www.eg.bucknell.edu/~phys310/jupyter/ode_sympy.html>
* Thanks to Oscar Benjamin on stackoverflow, <https://stackoverflow.com/a/68133782/16316043>, who helped me with SymPy.
* SymPy documentation: <https://docs.sympy.org/latest/index.html>

[notebook]: https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter/2021-06-27-Solving-the-schr%C3%B6dinger-equation-with-SymPy.ipynb


