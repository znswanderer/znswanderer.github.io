---
title:  "Reference sheet"
layout: home
---

## Momentum Operator

The operator for the linear momentum in position basis for the $$x$$-axis is 


$$
\hat p = -i \hbar \frac{\partial}{\partial x}
$$

The momentum operator is the [generator of spatial translation]({{site.url}}/physics/Momentum-Operator/).


## Time integration

The time-dependent Schrödinger equation is

$$
i \hbar \frac{d}{dt} | \psi(t) \rangle = \hat{H} | \psi(t) \rangle \, .
$$

For the time integration in python we will use a Runge-Kutta integrator from `scipy.integrate.solve_ivp`. This 
will solve an ODE of the form:

```
dy / dt = f(t, y)
y(t0) = y0
```

Here the ODE is the Schrödinger equation and the right hand side gives the python code

```python
f(t, psi) = (-1j / hbar) * (H @ psi)
```

So the integration is done in the following:

```python
t_eval = np.linspace(0, 1, 201)   # times from t=0 to t=1 with 201 steps
sol = scipy.integrate.solve_ivp(
    lambda t, psi: (-1j / hbar) * (H @ psi), 
    t_span = [t_eval[0], t_eval[-1]], 
    y0 = psi0, t_eval = t_eval, 
    method="RK23"    # Runge-Kutta integrator
)
```

where `psi0` is the `numpy` array representing $$\vert \psi(t=0) \rangle$$.
Then the solution `sol.y[psi_t, t]` is an array, that contains the 
wave function `psi_t` for every point in time `t`, that was specified in `t_eval`.

