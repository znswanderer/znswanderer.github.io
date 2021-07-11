---
layout: post
title:  "Yak shaving and more on the Harmonic Oscillator"
date:   2021-07-11
categories: blog
usemathjax: true
tags:   [Physics, Python, SymPy, Quantum Mechanics, Harmonic Oscillator, sympy.integrate, yak shaving]
---
Ok, this week I found myself engrossed in a bit of [yak shaving][yak]. It all started as I
wanted to use python functions as representations for quantum operators.

[yak]: https://en.wiktionary.org/wiki/yak_shaving
<!--more-->
In the first version of this python representation the Hamiltonian for the harmonic oscillator

$$
\hat H = \left( \hat a^\dagger \hat a + \frac{1}{2} \right)
$$

could be expressed as

{% raw %}
```python
Hamiltonian = add(mult(a_creation, a_annihilation), scalar_op(smp.Rational(1, 2)))
``` {% endraw %}

where for example the annihilation operator

$$
\hat a = \frac{1}{\sqrt{2}} \left( \hat x + i \hat p \right)
$$

would be abstractly defined in python as

{% raw %}
```python
a_annihilation = scalar_product(1/smp.sqrt(2), 
                                add(position, scalar_product(smp.I, momentum)))
``` {% endraw %}

and the position and momentum operators are then python functions, that take a `wav_func`
object as input, which is itself a SymPy function object:

{% raw %}
```python
def position(wav_func):
    return x * wav_func

def momentum(wav_func):
    return - smp.I * smp.diff(wav_func, x)
``` {% endraw %}

This was all very neat and (relatively) simple. 

## Operators as a Python Class

But then I thought, that all this
`add(op_A, op_B)` was not really "elegant" and I would like to do things more directly,
like adding operators: `op_A + op_B`. 

Python, like most programming languages, has support for overloading
arithmetic operations like `+` or `*` for self-defined types, via
the ["magic" methods][magic] like `__add__`, `__mul__` and so on.

So I quickly threw together a small script for trying this out:


[magic]: https://rszalski.github.io/magicmethods/#numeric


{% raw %}
```python
from numbers import Number
import sympy as smp

class Operator:

    number_types = (Number, smp.core.Expr)

    def __init__(self, func):
        self.func = func

    def __call__(self, v):
        return smp.simplify(self.func(v))

    def __mul__(self, other):
        if isinstance(other, Operator):
            return Operator(lambda v: self(other(v)))            
        elif isinstance(other, self.number_types):
            return Operator(lambda v: self(v) * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, self.number_types):
            return Operator(lambda v: other * self(v))
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, Operator):
            return NotImplemented
        elif isinstance(other, self.number_types):
            return Operator(lambda v: self(v)/other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Operator):
            return NotImplemented
        elif isinstance(other, self.number_types):
            return Operator(lambda v: self(v)/other)
        else:
            return NotImplemented        

    def __add__(self, other):
        if isinstance(other, Operator):
            return Operator(lambda v: self(v) + other(v))
        elif isinstance(other, self.number_types):
            return self + Scalar(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, self.number_types):
            return Scalar(other) + self

    def __sub__(self, other):
        if isinstance(other, Operator):
            return Operator(lambda v: self(v) - other(v))
        elif isinstance(other, self.number_types):
            return self - Scalar(other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, self.number_types):
            return Scalar(other) - self

    def __neg__(self):
        return Operator(lambda v: -self(v))

    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return Scalar(1)
            elif n == 1:
                return self
            elif n > 1:
                return self * self**(n - 1)
            else:
                return NotImplemented
        else:
            return NotImplemented


class Scalar(Operator):
    def __init__(self, s):
        self.s = s

    def __call__(self, v):
        return self.s * v


def canonical_operators(var):
    """Returns the canonical operators $$(\hat x, \hat p_x)$$ for the spatial dimension var
    in SymPy.
    """
    return Operator(lambda f: var * f), Operator(lambda f: - smp.I * smp.diff(f, var))
``` {% endraw %}

And it seemed to work quite well, for such a small script! 

First I defined the SymPy objects for the wave function $$\psi$$ and the 
spatial dimension $$x$$:


{% raw %}
```python
psi = smp.Function(r'\psi')
x = smp.symbols('x', real=True)
``` {% endraw %}

And now I can simply create the canonical operators for this dimension:


{% raw %}
```python
X, Px = canonical_operators(x)
``` {% endraw %}

With these operators it's rather simple to define the Hamiltion for the harmonic oscillator and
apply it to the wavefunction:


{% raw %}
```python
H = Px**2 / 2 + X**2 / 2
H(psi(x))
``` {% endraw %}
$$\displaystyle \frac{x^{2} \psi{\left(x \right)}}{2} - \frac{\frac{d^{2}}{d x^{2}} \psi{\left(x \right)}}{2}$$  



It is also quite easy to include more dimensions and get the Hamiltonian $$\hat H_{2d}$$ for the two-dimensional Oscillator:


{% raw %}
```python
y = smp.symbols('y', real=True)
Y, Py = canonical_operators(y)
H_2d = (Px**2 + Py**2) / 2 + (X**2 + Y**2) / 2
H_2d(psi(x, y))
``` {% endraw %}
$$\displaystyle \frac{\left(x^{2} + y^{2}\right) \psi{\left(x,y \right)}}{2} - \frac{\frac{\partial^{2}}{\partial x^{2}} \psi{\left(x,y \right)}}{2} - \frac{\frac{\partial^{2}}{\partial y^{2}} \psi{\left(x,y \right)}}{2}$$  



Or use the ladder operators:


{% raw %}
```python
ax = (X + smp.I * Px) / smp.sqrt(2)
ax_dag = (X - smp.I * Px) / smp.sqrt(2)
ay = (Y + smp.I * Py) / smp.sqrt(2)
ay_dag = (Y - smp.I * Py) / smp.sqrt(2)

(ax_dag * ax + ay_dag * ay)(psi(x, y))
``` {% endraw %}
$$\displaystyle \frac{x^{2} \psi{\left(x,y \right)}}{2} + \frac{y^{2} \psi{\left(x,y \right)}}{2} - \psi{\left(x,y \right)} - \frac{\frac{\partial^{2}}{\partial x^{2}} \psi{\left(x,y \right)}}{2} - \frac{\frac{\partial^{2}}{\partial y^{2}} \psi{\left(x,y \right)}}{2}$$  



By just comparing this expression with $$\hat H_{2d}$$ we can see, that

$$
\hat H_{2d} = \hat a_x^\dagger \hat a_x + \hat a_y^\dagger \hat a_y + 1
$$

So by just applying this simple operator framework we found that the ground energy of the harmonic oscillator in $$N_{dim}$$ 
[dimensions][dim_ho] is $$N_{dim}/2$$, which is not too surprising.

[dim_ho]: https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#N-dimensional_isotropic_harmonic_oscillator

## But then I started to get lost in the woods...

The little script above is quite neat, but it has a lot of shortcomings. So I departed on a sidequest
to make a *perfect* operator algebra python module, with vectors, dual vectos, bras and kets and Hilbert spaces and all the good stuff. 
Needless to say, in the beginning it seemed like a good idea, but the further I moved along, the more
it became clear, that I was in fact *shaving a yak.*

Why was I doing that? There are perfectly good libraries in Python available for these things. 
Libraries like:

* <https://docs.sympy.org/latest/modules/physics/quantum/index.html>
* <https://qutip.org/>

There's no need to re-invent the wheel!

And in the end, I will switch to numerics soon enough anyway. Then all the quantum stuff 
will be just matrix multiplication.

All very simple. And, as [someone][who] said: 

> Everything Should Be Made as Simple as Possible, But Not Simpler

Only the question remains: what is *simple?*

In the end, I shelved this "perfect" python module and will move on from symbolic/analytic treatment
of the harmonic oscillator with SymPy. But before that, there is a bonus track!

[who]: https://quoteinvestigator.com/2011/05/13/einstein-simple/

## Bonus Yak

Here we want to take a closer look at the first eigenfunctions of the harmonic oscillator. For this
we define two new useful operations in python: 

The normalisation of a wave function:

$$
\textrm{normalize}: \, \psi (x) \mapsto \frac{\psi (x)}{\int_{-\infty}^{\infty} |\psi(x)|^2 dx}
$$

and measuring an operator:

$$
\textrm{measure}: \, \left(\hat A, \psi(x)\right) \mapsto \int_{-\infty}^{\infty} \psi^*(x) \hat A \psi(x) dx
$$


{% raw %}
```python
def normalize(wav_func):
    """Normalize a wave function psi(x), such that
    $$\int_{-\infty}^\infty |psi(x)|^2 dx = 1
    """
    c = smp.integrate(smp.Abs(wav_func)**2, (x, -smp.oo, smp.oo))
    return wav_func / smp.sqrt(c)

def measure(operator, wav_func):
    """Compute <wav_func | operator | wav_func>
    """
    return smp.integrate(smp.conjugate(wav_func) * operator(wav_func), (x, -smp.oo, smp.oo))
``` {% endraw %}

As written [last time][last], the ground state for the harmonic oscillator can found by
the equation:

$$
\hat a \psi_0(x) = 0
$$

We once again use SymPy's `dsolve` for finding the solution and use the new operation
for normalisation:

[last]: https://znswanderer.github.io/blog/Harmonic-Oscillator-SymPy/


{% raw %}
```python
psi0 = normalize(smp.dsolve(ax(psi(x)), psi(x)).rhs)
psi0
``` {% endraw %}
$$\displaystyle \frac{C_{1} e^{- \frac{x^{2}}{2}}}{\sqrt[4]{\pi} \left|{C_{1}}\right|}$$  



Here SymPy reminds us, that the the wave function can be multiplied by a complex factor 
of unit length
$$C_1 / |C_1| = e^{i \alpha}, \, \alpha \in \mathbb{R}$$.
For simplicity we choose $$C_1 = 1$$:


{% raw %}
```python
C1 = smp.symbols('C1')
psi0 = psi0.subs([(C1, 1)])
``` {% endraw %}

And now we calculate the next eigenfunctions by repeated application[<sup>1</sup>](#fn1) of $$\hat a^\dagger$$:


{% raw %}
```python
from IPython.display import display, Math

N = 5
psi_ns = [psi0]
for n in range(1, N+1):
    f = smp.simplify(normalize(ax_dag(psi_ns[-1])))
    psi_ns.append(f)

lines = []
for n, f in enumerate(psi_ns):
    E = measure(H, f) # A quick check of the energy for these states
    lines.append(r"E_{%d} &= %s, \, \psi_{%d} = %s \\" % (n, smp.latex(E), n, smp.latex(f)))
    
display(Math("\\begin{aligned}\n%s\n\\end{aligned}" % "\n".join(lines)))
``` {% endraw %}
$$\displaystyle \begin{aligned}  
E_{0} &= \frac{1}{2}, \, \psi_{0} = \frac{e^{- \frac{x^{2}}{2}}}{\sqrt[4]{\pi}} \\
E_{1} &= \frac{3}{2}, \, \psi_{1} = \frac{\sqrt{2} x e^{- \frac{x^{2}}{2}}}{\sqrt[4]{\pi}} \\
E_{2} &= \frac{5}{2}, \, \psi_{2} = \frac{\sqrt{2} \left(x^{2} - \frac{1}{2}\right) e^{- \frac{x^{2}}{2}}}{\sqrt[4]{\pi}} \\
E_{3} &= \frac{7}{2}, \, \psi_{3} = \frac{\sqrt{3} x \left(2 x^{2} - 3\right) e^{- \frac{x^{2}}{2}}}{3 \sqrt[4]{\pi}} \\
E_{4} &= \frac{9}{2}, \, \psi_{4} = \frac{\sqrt{6} \left(\frac{x^{4}}{3} - x^{2} + \frac{1}{4}\right) e^{- \frac{x^{2}}{2}}}{\sqrt[4]{\pi}} \\
E_{5} &= \frac{11}{2}, \, \psi_{5} = \frac{\sqrt{15} x \left(4 x^{4} - 20 x^{2} + 15\right) e^{- \frac{x^{2}}{2}}}{30 \sqrt[4]{\pi}} \\
\end{aligned}$$


The energy eigenstates of the oscillator are not eigenstates of the position or momentum
operator. This means, in these states the position and the momentum will not be sharply defined.

So, it would be nice to take a look at the standard deviation of these operators.
The standard deviation of an operator $$\hat A$$ for the system in state $$\ket \psi$$ is
defined as:

$$
\left( \Delta A \right)_\psi = \sqrt{
    \bra \psi A^2 \ket \psi - \bra \psi A \ket \psi^2
}
$$

With the position space wave function $$\psi(x)$$ we can define this in python as:


{% raw %}
```python
def delta(operator, wav_func):
    return smp.sqrt(measure(operator**2, wav_func) - measure(operator, wav_func)**2)
``` {% endraw %}

First, let's check if the deviation of the energy really vanishes for the eigenfunctions:


{% raw %}
```python
lines = []
for n, f in enumerate(psi_ns):
    lines.append(r"{\Delta E}_%d &= %s \\" % (n, smp.latex(delta(H, f))))
    
display(Math("\\begin{aligned}\n%s\n\\end{aligned}" % "\n".join(lines)))
``` {% endraw %}
$$\displaystyle \begin{aligned}  
{\Delta E}_0 &= 0 \\
{\Delta E}_1 &= 0 \\
{\Delta E}_2 &= 0 \\
{\Delta E}_3 &= 0 \\
{\Delta E}_4 &= 0 \\
{\Delta E}_5 &= 0 \\
\end{aligned}$$


Now, let's check for the famous expression $$\Delta x \Delta p$$:


{% raw %}
```python
lines = []
for n, f in enumerate(psi_ns):
    dx = delta(X, f)
    dp = delta(Px, f)
    lines.append(r"\left(\Delta x \Delta p \right)_{} &= {} \\".format(
        n, smp.latex(dx * dp)))
    
display(Math("\\begin{aligned}\n%s\n\\end{aligned}" % "\n".join(lines)))
``` {% endraw %}
$$\displaystyle \begin{aligned}  
\left(\Delta x \Delta p \right)_0 &= \frac{1}{2} \\
\left(\Delta x \Delta p \right)_1 &= \frac{3}{2} \\
\left(\Delta x \Delta p \right)_2 &= \frac{5}{2} \\
\left(\Delta x \Delta p \right)_3 &= \frac{7}{2} \\
\left(\Delta x \Delta p \right)_4 &= \frac{9}{2} \\
\left(\Delta x \Delta p \right)_5 &= \frac{11}{2} \\
\end{aligned}$$


So, if we re-introduce $$\hbar$$, the ground state has the lowest possible uncertainty
according to the uncertainty principle. And the overall uncertainty grows proportionally
with the energy in the harmonic oscillator. An analytic derivation of this series
can be found on the corresponding [wikipedia] page on the harmonic oscillator.

[wikipedia]: https://en.wikipedia.org/wiki/Uncertainty_principle#Quantum_harmonic_oscillator_stationary_states

## Final remarks

Ok, that's all for now. There is still much, that can be done by just using 
SymPy in the study of quantum mechanics. But I think, it's time for moving 
to numerics and linear algebra in the next blog post.

### Footnotes

<span id="fn1">1: This is surely not the most efficient way to get these functions.
    For example in the `sympy.physics` module one can find all 
    eigenfunctions with `qho_1d.psi_n()`.
</span>

*The original Jupyter notebook can be found [here](<https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter//2021-07-11-Yak-Shaving-Oscillator.ipynb>).*