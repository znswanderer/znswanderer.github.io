---
layout: post
title:  "The Angular Momentum as Generator of Rotation"
date:   2022-09-01
categories: Physics
tags:   [Quantum Mechanics, Angular Momentum]
---
{% raw %}
In this post we will examine how to perform a rotation in a two-dimensional system. We will see, that 
the rotation is linked to the angular momentum operator, just like the spatial translation is generated
by the [momentum operator](https://znswanderer.github.io/physics/Momentum-Operator/).

A wavefunction $$\psi(x,y)$$ in the $$xy$$-plane shall be rotated by an angle $$\Delta \varphi$$
with respect to the origin, resulting in a new wavefunction $$\tilde \psi(x)$$ (not shown
in the picture for clarity).

{: style="text-align:center"}
{% endraw %}
![Image]({{site.url}}/assets/images/2022-09-03-Angular-Momentum_files/rotation_sep2022.png)
{% raw %}

{% endraw %}
<!--more-->
{% raw %}

If we want to rotate a point in the $$xy$$-plane given by the vector $$\vec p = \begin{bmatrix} x & y \end{bmatrix}^T$$ 
we can do this by applying a [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix):

$$
\vec p' =
\begin{bmatrix}
x' \\
y'
\end{bmatrix} 
=
\begin{bmatrix}
\cos \Delta \varphi & -\sin \Delta \varphi \\
\sin \Delta \varphi & \cos \Delta \varphi
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

(Here we are using square brackets for vectors and matrices to distinguish these objects from the vectors and operators
that we use for quantum wave functions and operators, for which we use round brackets)

The problem now is that we want to rotate not only single points, but the complete set of values for the entire $$xy$$-plane, 
i.e. the wave function $$\varphi(x,y)$$ for all $$x$$ and $$y$$ at the same time.
We do this by focusing on the operation itself, that moves a point $$\vec p$$ to $$\vec p'$$:

$$
\Delta \vec r =
\vec p' - \vec p 
=
\begin{bmatrix}
0 & -\sin \Delta \varphi \\
\sin \Delta \varphi & 0
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

or for small $$\Delta \varphi$$

$$
\Delta \vec r \approx
\left(- y \vec e_x + x \vec e_y \right)   \Delta \varphi 
$$

This can be understood as two separate translations, one in the $$x$$-direction and one in the $$y$$-direction:
$$\Delta \vec r = \Delta \vec r_x + \Delta \vec r_x$$ with

$$
\begin{align}
\Delta \vec r_x &= -y \Delta \varphi \vec e_x \\
\Delta \vec r_y &= x \Delta \varphi \vec e_y \, .
\end{align}
$$

## Moving to quantum mechanics

So basically we have two separate translations for the $$x$$ and $$y$$ direction and the amount
of the translation is modified by $$\Delta \varphi$$ and the distance to the origin in the complementary coordinate 

$$
\begin{aligned}
U_{\Delta x}:& \quad
\Delta x = -y \Delta \varphi \\
U_{\Delta y}:& \quad\Delta y = x \Delta \varphi
\end{aligned}
$$

Imagine a state completely localized at $$x$$ and $$y$$, that is

$$
|x, y \rangle = |x \rangle \otimes |y \rangle \, .
$$

The distance from the origins can be retrieved by the $$\hat x$$ and $$\hat y$$ operators as the localized states
are eigenfunctions of the operators

$$
\begin{aligned}
x |x \rangle &= \hat x |x \rangle \\
y |y \rangle &= \hat y |y \rangle \, .
\end{aligned}
$$

As we know from the [last post](https://znswanderer.github.io/physics/Momentum-Operator/), the operation of a translation is done by an unitary operator, 
whose generator is the momentum operator. So if we translate this localised wavefunction by $$\Delta \vec r$$ we find

$$
|\tilde x, \tilde y \rangle  =
\left( \hat U_{-y \Delta \varphi} \otimes \hat U_{x \Delta \varphi} \right)
\left( |x \rangle \otimes |y \rangle \right)
$$

If we keep $$\Delta \varphi$$ small, such that both $$-y \Delta \varphi$$ and $$x \Delta \varphi$$ are small we can write

$$
\begin{aligned}
|\tilde x, \tilde y \rangle  &\approx
\left(
\left[ 1 +  i y \Delta \varphi \hat p_x  / \hbar \right]
\otimes
\left[ 1 -  i x \Delta \varphi \hat p_y  / \hbar \right]
\right)
|x, y \rangle \\
&\approx
\left( 1
-  i \left[ x \otimes \hat p_y \right] \Delta \varphi / \hbar
+  i \left[ \hat p_x \otimes y \right] \Delta \varphi / \hbar
+ \mathcal{O}(\Delta \varphi^2)
\right)
|x, y \rangle
\end{aligned}
$$

The scalars $$x$$ and $$y$$ in the expression are just the eigenvalues for the localized
state $$\vert x, y \rangle$$ so we can insert the position operators

$$
|\tilde x, \tilde y \rangle  \approx
\left(
1 - i\left[
\hat x \otimes \hat p_y - \hat p_x \otimes \hat y
\right] \Delta \varphi / \hbar
\right)
|x, y \rangle
$$
(Ignoring terms $$\mathcal{O}(\Delta \varphi^2)$$ for small $$\Delta \varphi$$)

The inner operator is the [*angular momentum* operator](https://en.wikipedia.org/wiki/Angular_momentum_operator) (for the $$xy$$-plane)

$$
\boxed{
\hat L = \hat x \otimes \hat p_y - \hat p_x \otimes \hat y
}
$$

with this

$$
|\tilde x, \tilde y \rangle  \approx
\left(
1 - i \hat L \Delta \varphi / \hbar
\right)
|x, y \rangle \, .
$$

As the position and momentum operators are Hermitian, so is $$\hat L$$, which makes sense as the angular momentum
is an observable.

The derivation above was done for a localized wavefunction at $$(x, y)$$. By using 
the position basis for $$x$$ and $$y$$, we can imagine every wavefunction as a combination of localized wavefunctions, 
$$\vert \psi \rangle = \int \psi(x,y) \vert x, y\rangle dx dy$$. Operators in quantum mechanics are *linear* operators,
so if $$\hat L$$ can be used to rotate $$\vert x, y \rangle$$ so it can be used for a linear 
combination of position space vectors. So the infinitesimally rotated wavefunction $$\vert \tilde \psi \rangle$$ is

$$
|\tilde \psi \rangle  \approx
\left(
1 - i \hat L \Delta \varphi / \hbar
\right)
| \psi \rangle \, .
$$

Now, just like we did for the [translation operator](https://znswanderer.github.io/physics/Momentum-Operator/),
we can model a non-infinitesimal rotation as a repeated rotation by infinitesimal amounts and in the limit get the
rotation operator

$$
\hat U_{\Delta \varphi} = \exp (-i \Delta \varphi \hat L / \hbar ) \, .
$$

## Spin

In quantum mechanics we find something, that has no classical counterpart. There is a notion of an internal degree
of freedom called *spin*. The complete [generator of rotation](https://en.wikipedia.org/wiki/Angular_momentum_operator#Angular_momentum_as_the_generator_of_rotations) is

$$
\hat J = \hat L + \hat S
$$

which includes the contribution, $$\hat S$$, due to spin. In the following we will ignore spin.


## Angular Momentum in polar coordinates

In position space the operator of angular momentum becomes

$$
\hat L \psi(x, y) = -i \hbar \left( x \frac{\partial}{\partial y} - y \frac{\partial}{\partial x} \right) \psi(x,y)
$$

In polar coordinates we have

$$
\begin{align}
x &= r \cos \varphi \\
y &= r \sin \varphi \, .
\end{align}
$$

Changing the variables for the differentiation we have to find

$$
\begin{align}
\frac{\partial}{\partial x} &= \frac{\partial r}{\partial x} \frac{1}{\partial r} 
                                + \frac{\partial \varphi}{\partial x} \frac{1}{\partial \varphi} \\
\frac{\partial}{\partial y} &= \frac{\partial r}{\partial y} \frac{1}{\partial r} 
                                + \frac{\partial \varphi}{\partial y} \frac{1}{\partial \varphi} \, .
\end{align}
$$

After some calculation we find

$$
\begin{align}
\frac{\partial r}{\partial x} = \cos \varphi    \qquad \qquad   &\frac{\partial \varphi}{\partial x} = - \frac{\sin \varphi}{r} &\\
\frac{\partial r}{\partial y} = \sin \varphi    \qquad \qquad   &\frac{\partial \varphi}{\partial y} =  \frac{\cos \varphi}{r} &
\end{align}
$$

Inserting these terms into the equation for $$\hat L$$ in the cartesian coordinates this gives

$$
\hat L \psi(r, \varphi) = - i \hbar \frac{\partial}{\partial \varphi} \psi(r, \varphi) \, ,
$$
because the terms involving $$\partial_r$$ cancel each other out. So only the dependency on the angle
defines the angular momentum.

### Rotational symmetry and Eigenvalues

Suppose we have an eigenstate of the angular momentum operator

$$
\hat L |\phi \rangle = \lambda |\phi \rangle \, .
$$

As $$\hat L$$ is Hermitian the eigenvalue is reel, $$\lambda \in \mathbb{R}$$.

If we rotate the eigenstate we find

$$
|\tilde \phi \rangle = \hat U_{\Delta \varphi} |\phi \rangle = e^{-i \Delta \varphi \hat L / \hbar} |\phi\rangle = e^{-i \lambda \Delta \varphi} |\phi\rangle \, .
$$

So the rotation results only in a constant unitary factor
and the probabiltity densitiy of the eigenstate is unchanged by the rotation:

$$
\langle \tilde \phi | \tilde \phi \rangle = \langle \phi | \phi \rangle \, ,
$$

which means, that the probability densitiy of these states does not depend on the angle.

States, that have a rotational symmetry in the density w.r.t. the origin have
no dependency on $$\varphi$$ in the probability density:

$$
|\psi(r, \varphi)|^2 = |\psi(r)|^2
$$


such states can be described by the product

$$
\psi(r, \varphi) = \rho(r) e^{i m \varphi} \, .
$$

As $$\psi(r, \varphi)$$ must be continuous, that means $$\psi(r, \varphi + 2\pi) = \psi(r, \varphi)$$,
it follows

$$
e^{2 i  m \pi} = 1
$$

so we have the allowed values $$m \in \mathbb{Z}$$.

With the operator of angular momentum in polar coordinates it is easy to see, that these rotational symmetric 
functions are also eigenfunctions of $$\hat L$$:

$$
\hat L  e^{i m \varphi} = - i  \hbar \frac{\partial}{\partial \varphi} e^{i m \varphi} = m\hbar e^{i m \varphi} \, .
$$

So we found eigenfunctions of $$\hat L$$ with the eigenvalues $$\lambda_m = m \hbar$$.

### Problematic Eigendecomposition and Schrödinger-like differential equation

The angular momentum operator only acts on the dimension of the wave function, that is connected
to the polar angle $$\varphi$$. That means, that we can have two wave functions, $$\psi_1(r, \varphi) = \rho_1(r) e^{im\varphi}$$
and $$\psi_2(r, \varphi) = \rho_2(r) e^{im\varphi}$$, that are eigenstates of $$\hat L$$ with eigenvalue $$m$$, but with
different radial dependency $$\rho_1(r) \neq \rho_2(r)$$. Therefore we cannot use the eigenstates of the angular momentum alone 
to construct the identity matrix

$$
\mathbf{1} \neq \sum_m |m\rangle \langle m | \, ,
$$

but we need additional "information" about a quantum state, for example an energy eigenstate 
$$\hat H \vert n \rangle = E_n \vert n \rangle$$ (if the Hamiltonian has rotational symmetry). 
A sufficient set of eigenvalues for describing a quantum state uniquely is called a 
[complete set of commuting variables](https://en.wikipedia.org/wiki/Complete_set_of_commuting_observables).

Having no complete basis involving the angular momentum operator, we cannot use the eigendecomposition 
of $$e^{-i \Delta \varphi \hat L / \hbar}$$ like we did in the [last post](https://znswanderer.github.io/physics/Momentum-Operator/)
for the spatial translation. But there is another way to use the angular momentum operator for a rotation operation.
We can describe the rotation of a state $$\vert \psi_0 \rangle$$ as a continuous process w.r.t. the angle $$\varphi$$:

$$
|\psi_\varphi \rangle = e^{-i \varphi \hat L / \hbar} |\psi_0 \rangle \, .
$$

Now we can differentiate by $$\varphi$$ and get

$$
\frac{d}{d \varphi} |\psi_\varphi \rangle = \frac{-i \hat L}{\hbar} e^{-i \varphi \hat L / \hbar} |\psi_0 \rangle
$$

which can be written as

$$
\boxed{
i \hbar \frac{d}{d \varphi} \vert \psi_\varphi \rangle= \hat L \vert \psi_\varphi \rangle
}
$$

This really looks like the time dependent Schrödinger equation and it becomes more evident, why $$\hat L$$ is called
the *generator* of rotation. On the practical side we can use this diffential equation as the starting point for a numerical integration,
which will be done in the code example.

# Code example

In the post [Quantum Systems in two Spatial Dimensions](https://znswanderer.github.io/physics/2d-Quantum-Systems/)
we have described how to construct a two dimensional system:


```python
import numpy as np
from types import SimpleNamespace
from scipy import integrate, sparse
import matplotlib.pyplot as plt
from IPython.display import Video

def make_2d_box(N, L=2):
    L = L
    hbar = 1
    mass = 1
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    dx = np.diff(x)[0]
    One = sparse.eye(N)

    return SimpleNamespace(**locals())

box = make_2d_box(N=300)
```

For the momentum operator in the definition of the angular momentum we will use the central difference operator:

$$
\frac{d}{dx} f(x) = \frac{1}{2 \Delta x} \left( f(x + \Delta x) - f(x - \Delta x) \right) + \mathcal{O}({\Delta x}^3)
$$


```python
def angular_momentum(box):
    x_op = sparse.kron(sparse.diags([box.x], [0]), box.One)
    y_op = sparse.kron(box.One, sparse.diags([box.y], [0]))
    D_s_op = sparse.diags([-1, 1], [-1, 1], shape=(box.N, box.N)) / (2*box.dx)
    p_op = - 1j * box.hbar * D_s_op
    px_op = sparse.kron(p_op, box.One)
    py_op = sparse.kron(box.One, p_op)
    L_op = x_op @ py_op - px_op @ y_op
    return L_op

L_op = angular_momentum(box)
```

Let's use a gaussian wave packet at $$x_0 = 0.5$$ and $$y_0 = 0$$ for the "rotation experiment"


```python
sigma = 0.1
gauss = 1j * np.kron(np.exp(-(box.y)**2/sigma), np.exp(-(box.x-0.5)**2/sigma))
gauss_2d = np.reshape(gauss, (box.N, box.N))
plt.imshow(np.abs(gauss_2d)**2, extent=[box.x[0], box.x[-1], box.y[0], box.y[-1]], cmap='gnuplot2');
```


    
{: style="text-align:center"}
{% endraw %}
![Image]({{site.url}}/assets/images/2022-09-03-Angular-Momentum_files/2022-09-03-Angular-Momentum_25_0.png)
{% raw %}
    


The $$\varphi$$ integration will be done using the `scipy.integrate.solve_ivp` function, just like
we did for the time integration in a [previous post](https://znswanderer.github.io/physics/Time-Evolution/).


```python
phi_eval = np.linspace(0, 2*np.pi, 201)
sol = integrate.solve_ivp(lambda phi, psi: (-1j / box.hbar) * (L_op @ psi), 
                          t_span = [phi_eval[0], phi_eval[-1]], y0 = gauss, t_eval = phi_eval, method="RK23")
```

(The class WaveAnimation can be found in the notebook)


```python
WaveAnimation(psi_t=sol.y, t_vec=sol.t, box=box).save_mp4('gauss_rotated.mp4')
Video('gauss_rotated.mp4', width=600)
```




{% endraw %}
<video src="{{site.url}}/assets/images/2022-09-03-Angular-Momentum_files/gauss_rotated.mp4" controls  width="600" >
{% raw %}
      Your browser does not support the <code>video</code> element.
    </video>



# Epilog

Moving a state through time, translating a distance or rotating by an angle,
are similar transformations, that can be described by
different kind of "Schrödinger equations".
For example, in a translation of $$x$$ we have:

$$
|\psi_x\rangle = e^{-i x \hat p / \hbar} |\psi_0 \rangle
$$

Note that this is not the position space wave function $$\psi(x)$$, but the ket-vector of the state
$$\vert \psi_0 \rangle$$ shifted by an amount of $$x$$. The variable $$x$$ is not the dimesion in space, but
the (scalar) shift along this direction.

If we differentiate by $$x$$ we get

$$
\frac{d}{dx} |\psi_x \rangle = -\frac{i \hat p}{\hbar} e^{-i x \hat p / \hbar} |\psi_0 \rangle
= -\frac{i \hat p}{\hbar} |\psi_x \rangle
$$

We find similar Schrödinger-like differential equations for different operations:

| Operation |  |
|------|------|
| Shift in Time  | $$i \hbar \frac{d}{dt} \vert \psi_t \rangle = \hat H \vert \psi_t \rangle$$ |
| Translation  | $$i \hbar \frac{d}{dx} \vert \psi_x \rangle= \hat p \vert \psi_x \rangle$$ |
| Rotation  | $$i \hbar \frac{d}{d \varphi} \vert \psi_\varphi \rangle= \hat L \vert \psi_\varphi \rangle$$ |

*The original Jupyter notebook can be found [here](<https://github.com/znswanderer/znswanderer.github.io/blob/main/_jupyter/2022-09-03-Angular-Momentum.ipynb>).*
 {% endraw %}