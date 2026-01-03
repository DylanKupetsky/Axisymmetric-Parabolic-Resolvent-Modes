# Axisymmetric Parabolic Resolvent Modes
Work for ESCI-995 (Numerical Ocean Circulation modeling) project inspired by work on identifying streaky structures in boundary layers by [Sasaki, Kenzo and Cavalieri](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.7.104611). Project as part of a larger effort to understand wake breakdown/interaction in floating offshore windfarm wake arrays. Here, using an inner product defined in cylindrical coordinates, and develop new adjoint equations to identify streaky structures behind an axisymmetric wake. 

Streaks play a significant role in transition to turbulence in boundary layers, as their destruction (a so-called "bursting event") generates significant turbulent kinetic energy. Resolvent analysis is an appropriate tool for study of streak formation, as development is considered non-modal. 

The inner product used is

$$
{\langle q, q \rangle_q = \int_0^{x_f}\int_0^{\infty} \mathbf{\overline{q}}\mathbf{q}rdrdx}
$$

and

$$
{\nabla \cdot \mathbf{u} = 0}
$$

$$
{\partial_t \mathbf{u'} + W\partial_z \mathbf{u'} + \nabla p' - \frac{1}{Re_\delta}\nabla^2 \mathbf{u'} = \mathbf{f'}}
$$

which are Orr-Somerfeld-like equations in axisymmetric coordinates, derived from a boundary-layer like asymptotic reduction, with additional forcings representing non-linear q'q' interactions.

The axial base flow, W(r,z), is a wake taken from a seperate simulation whose data can be found at this link.

We perform a Fourier transform in the time and azimuthal coordinates of the forcings and responses:

$$
\mathbf{f'}(r,\phi,z,t) = \int_0^{2\pi} \int_{-\infty}^{\infty} \mathbf{\hat{f}}(r,z)dm d\omega
$$

$$
\mathbf{q'}(r,\phi,z,t) = \int_0^{2\pi} \int_{-\infty}^{\infty} \mathbf{\hat{q}}(r,z)dm d\omega.
$$

and maximize the gain according to the numerical procedure outlined in [Sasaki, Kenzo and Cavalieri](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.7.104611). In our case, the adjoint equations are different, owing to the addition weight "r" appearing in the inner product.
