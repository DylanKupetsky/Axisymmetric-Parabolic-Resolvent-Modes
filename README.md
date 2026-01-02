# Axisymmetric-Parabolic-Resolvent-Modes
Work for ESCI-995 (Numerical Ocean Circulation modeling) project inspired by work on identifying streaky structures in boundary layers. Here, using an inner product defined in cylindrical coordinates, and develop new adjoint equations to identify streaky structures behind an axisymmetric wake.

The inner product used is

<mark></mark>
<mark>![equation](https://latex.codecogs.com/svg.image?%5Cint_%7B0%7D%5E%7Bx_f%7D%5Cint_%7B0%7D%5E%7B%5Cinfty%7D%5Cmathbf%7B%5Coverline%7Bq%7D%7D%5Cmathbf%7Bq%7Drdrdx)</mark>

and the linearized cylindrical incompressible Navier-Stokes equations

![equation](https://latex.codecogs.com/svg.image?%5Cnabla%5Ccdot%5Cmathbf%7Bu%7D=0)

![equation](https://latex.codecogs.com/svg.image?%5Cpartial_t%5Cmathbf%7Bu%7D&plus;W%5Cpartial_z%5Cmathbf%7Bu%7D&plus;%5Cnabla%20p-%5Cfrac%7B1%7D%7BRe_%7B%5Cdelta%7D%7D%5Cnabla%5E2%5Cmathbf%7Bu%7D=0)
