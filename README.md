# Axisymmetric Parabolic Resolvent Modes
Work for ESCI-995 (Numerical Ocean Circulation modeling) project inspired by work on identifying streaky structures in boundary layers. Here, using an inner product defined in cylindrical coordinates, and develop new adjoint equations to identify streaky structures behind an axisymmetric wake.

The inner product used is


![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BTeal%7D%5Cint_%7B0%7D%5E%7Bx_f%7D%5Cint_%7B0%7D%5E%7B%5Cinfty%7D%5Coverline%7B%5Cmathbf%7Bq%7D%7D%5Cmathbf%7Bq%7Drdrdx%7D)

and the linearized cylindrical incompressible Navier-Stokes equations

![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BTeal%7D%5Cnabla%5Ccdot%5Cmathbf%7Bu%7D=0%7D)

![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BTeal%7D%5Cpartial_t%5Cmathbf%7Bu%7D&plus;W%5Cpartial_z%5Cmathbf%7Bu%7D&plus;%5Cnabla%20p-%5Cfrac%7B1%7D%7BRe_%5Cdelta%7D%5Cnabla%5E2%5Cmathbf%7Bu%7D=0%7D)

which are Orr-Somerfeld-like equations in axisymmetric coordinates. The axial base flow, W(r,z), is a wake taken from a seperate simulation whose data can be found at this link.

We perform a Fourier transform in the time and azimuthal coordinates

![equation](https://latex.codecogs.com/svg.image?%7B%5Ccolor%7BTeal%7D%5Cmathbf%7Bq%7D(r,%5Cphi,z,t)=%5Cint_%7B-%5Cinfty%7D%5E%7B%5Cinfty%7D%5Cint_%7B0%7D%5E%7B2%5Cpi%7D%5Chat%7B%5Cmathbf%7Bq%7D%7D(r,z)%5Cexp(i(m%5Cphi-%5Comega%20t))d%5Comega%20dm%7D)
