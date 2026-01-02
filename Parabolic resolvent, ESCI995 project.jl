# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Julia 1.10.5
#     language: julia
#     name: julia-1.10
# ---

# %%
using Plots
using LinearAlgebra
using HDF5

# %%
#Build fourth order finite difference matrices 
#Which uses various schemes at the edges


using LinearAlgebra, SparseArrays

"""
Returns a vector `w` of weights such that
    f^(m)(0) ≈ sum_j w[j] * f(x_j)
where x_j = offsets[j]*h.
"""

function fd_weights(offsets::Vector{Int}, m::Int, h::Float64)
    n = length(offsets)
    # x positions
    x = [o * h for o in offsets]
    # Build Vandermonde-like matrix: M[k+1, j] = x_j^k  for k = 0 .. n-1
    M = Array{Float64}(undef, n, n)
    for k in 0:(n-1)
        for j in 1:n
            M[k+1, j] = x[j]^k
        end
    end
    # RHS: b_k = k! if k == m else 0  (k = 0..n-1)
    b = zeros(Float64, n)
    if m <= n-1
        b[m+1] = factorial(m)
    else
        error("Derivative order m must be <= number of stencil points - 1")
    end
    # Solve M * w = b
    w = M \ b
    return w
end

"""
Return sparse matrices (Dr, Drr) of size NxN approximating first and second derivatives
with global 4th-order accuracy on a uniform grid with spacing h.

Boundary rows are computed using one-sided 4th-order stencils (forward/backward).
Interior uses centered 5-point (offsets -2,-1,0,1,2).
"""

function build_fd_matrices(N::Int, h::Float64)
    @assert N >= 5 "Need at least 5 grid points for 4th-order scheme"

    rows = Int[]
    cols = Int[]
    vals1 = Float64[]   # for first derivative
    vals2 = Float64[]   # for second derivative

    function place_stencil!(i, offsets::Vector{Int})
        for (k,off) in enumerate(offsets)
            j = i + off
            if j < 1 || j > N
                error("Stencil index out of bounds: i=$i off=$off (N=$N).")
            end
            push!(rows, i)
            push!(cols, j)
            # weights will be added later for each derivative
        end
    end
    
    for i in 1:N
        # choose stencil offsets:
        if 3 <= i <= N-2
            # interior: centered 5-point offsets -2,-1,0,1,2
            offsets = [-2, -1, 0, 1, 2]
        elseif i == 1
            offsets = [0, 1, 2, 3, 4]      # forward one-sided at left boundary
        elseif i == 2
            offsets = [-1, 0, 1, 2, 3]     # near-left: mostly forward
        elseif i == N-1
            offsets = [-3, -2, -1, 0, 1]   # near-right: mostly backward
        elseif i == N
            offsets = [-4, -3, -2, -1, 0]  # backward one-sided at right boundary
        else
            # fallback (shouldn't reach)
            offsets = [-2, -1, 0, 1, 2]
        end

        # compute weights for first and second derivative
        w1 = fd_weights(offsets, 1, h)   # weights already scaled for h
        w2 = fd_weights(offsets, 2, h)

        # append entries for this row
        for (k,off) in enumerate(offsets)
            j = i + off
            push!(rows, i)
            push!(cols, j)
            push!(vals1, w1[k])
            push!(vals2, w2[k])
        end
    end

    # Now build sparse matrices.
    Dr = sparse(rows, cols, vals1, N, N)
    Drr = sparse(rows, cols, vals2, N, N)

    return Dr, Drr
end

r = collect(range(0,5,200));
dr = r[2] - r[1];
Dr, Drr = build_fd_matrices(200,r[2]-r[1]);

# %%
W = h5open("W_array.h5", "r")
Warray = W["W_array"]

# %%
test = real.(Warray[1:200,1])
Warray = real.(Warray[1:200,:])
# plot(real.(r),real.(test))
# plot!(real.(r),real.(Dr*test)) #Test the radial derivative to make sure it looks reasonable

# %%
z = collect(range(0,50,13400))
Δz = z[2] - z[1]
Wz = Wz = (Warray[:, 2:end] .- Warray[:, 1:end-1]) ./ Δz;
dZ = Δz;

# %%
#Build block matrices (forward and adjoint)

##The below code was the original code written to work only for m = 0. For m = 1, 2 implementation, see further down in the notebook.

function L(rvec,Dr,Drr,W,dZ,omega,Nr, Red)
    rvec[1] = 0.01 #prevents blow up for 1/r at  r= 0
    
    rmat = Diagonal(rvec)
    oneonrmat = Diagonal(1 ./rvec)
    Wmat = Diagonal(W)
    Wrmat = Diagonal(Dr*W)
    id = Matrix{ComplexF64}(I, Nr, Nr)

    Lmat = zeros(ComplexF64, 3*Nr, 3*Nr)

    eq_cont = 1:Nr
    eq_ax   = Nr+1:2*Nr
    eq_rad  = 2*Nr+1:3*Nr

    var_u = 1:Nr
    var_w = Nr+1:2*Nr
    var_p = 2*Nr+1:3*Nr

    Lmat[eq_cont, var_u] .= Dr .+ oneonrmat
    Lmat[eq_cont, var_w] .= 1/dZ #since we have w_{j + 1} - w_{j} / dZ, LHS is just w_{j + 1}/dZ and RHS is w_{j}/dZ (see function "b")

    

    Lmat[eq_ax, var_u] .= Wrmat
    Lmat[eq_ax, var_w] .= -1im*omega*id .+ Wmat/dZ .- (1/Red)*(Drr .+ oneonrmat.*Dr)

    Lmat[eq_rad, var_u] .= -1im*omega*id .+ Wmat/dZ .- (1/Red)*(Drr .+ oneonrmat.*Dr .- (oneonrmat*oneonrmat))
    Lmat[eq_rad, var_p] .= Dr

    #Set dirichlet conditions on u
    #Code works much better if actually using neumann at the edge... why? I believe fluid is being entrained at z = 0 farther than r/δ = 5, so we need to allow fluid to flow in from outside the domain.
    Lmat[1,:] .= 0
    Lmat[1,1] = 1
    Lmat[eq_cont[end],:] .= 0
    # Lmat[eq_cont[end],var_u[end]] = 1
    Lmat[eq_cont[end],var_u] .= Dr[end,1:Nr]

    #set neumann condition on w and dirichlet at edge
    Lmat[eq_ax[1],:] .= 0
    Lmat[eq_ax[1],var_w] .= Dr[1,1:Nr]
    Lmat[eq_ax[end],:] .= 0
    Lmat[eq_ax[end],var_w[end]] = 1

    #set neumann condition on p and pressure constraint
    Lmat[eq_rad[end],:] .= 0
    Lmat[eq_rad[1],:] .= 0
    Lmat[eq_rad[1],var_p] .= Dr[1,1:Nr]
    Lmat[eq_rad[end],var_p[1]] = rvec[1]*dr*.5
    Lmat[eq_rad[end],var_p[end]] = rvec[end]*dr*.5 #impose ∫ p r dr  = 0 using trapezoid
    return Lmat
end


function b(W,wprev,uprev,fs,fd,dZ,rvec)
    bmat = zeros(ComplexF64,3*Nr)
    Wmat = Diagonal(W)
    
    eq_cont = 1:Nr
    eq_ax   = Nr+1:2*Nr
    eq_rad  = 2*Nr+1:3*Nr
    
    bmat[eq_cont] .= wprev/dZ
    # radial bcs
    bmat[1] = 0.0
    bmat[eq_cont[end]] = 0.0

   
    bmat[eq_ax] .= Wmat*wprev/dZ .+ fs
    #axial bcs
    bmat[eq_ax[1]] = 0.0
    bmat[eq_ax[end]] = 0.0

    bmat[eq_rad] .= Wmat*uprev/dZ .+ fd
    #pressure regularization
    bmat[eq_rad[1]] = 0.0
    bmat[eq_rad[end]] = 0.0

    return bmat
end



function L_adj(rvec,Dr,Drr,W,Wz,dZ,omega,Nr, Red)
    rvec[1] = 0.01 #prevents infinity at r = 0
    dr = rvec[2] - rvec[1]
    Weight = dr*dZ*Diagonal(vcat(rvec, rvec, rvec))
    Winv   = Diagonal(1.0 ./ Weight)
    
    rmat = Diagonal(rvec)
    oneonrmat = Diagonal(1 ./rvec)
    Wmat = Diagonal(W)
    Wrmat = -Wmat*Dr .- oneonrmat*Wmat #adjoint of DrW
    Wzmat = Diagonal(Wz)
    id = Matrix{ComplexF64}(I, Nr, Nr)

    #form blocks by looking at full operators
    Astar = zeros(ComplexF64, 3*Nr, 3*Nr)
    Bstar = zeros(ComplexF64, 3*Nr, 3*Nr)
    Cstar = zeros(ComplexF64, 3*Nr, 3*Nr)
    Cstar_drr = zeros(ComplexF64, 3*Nr, 3*Nr)
    Bstar_dr = zeros(ComplexF64, 3*Nr, 3*Nr)
    Cstar_dr = zeros(ComplexF64, 3*Nr, 3*Nr)
    
    Estar = zeros(ComplexF64, 3*Nr, 3*Nr)
    Brstar = zeros(ComplexF64, 3*Nr, 3*Nr)
    Exstar = zeros(ComplexF64, 3*Nr, 3*Nr)

    eq_cont = 1:Nr 
    eq_ax   = Nr+1:2*Nr
    eq_rad  = 2*Nr+1:3*Nr

    var_us = 1:Nr
    var_ws = Nr+1:2*Nr
    var_ps = 2*Nr+1:3*Nr

    #Adjoint operator is (A* - B*/r - ∂_r B* - ∂_x E*) + C* ∂_rr - (B*∂_r - C*∂_r) + Estar ∂_x

    Astar[eq_cont,var_us] .= oneonrmat
    Astar[eq_cont,var_ws] .= Wrmat
    Astar[eq_cont,var_ps] .= 1im*omega .+ (1/Red)*oneonrmat*oneonrmat
    Astar[eq_ax,var_ws] .= 1im*omega

    Bstar[eq_cont,var_us] .= oneonrmat
    Bstar[eq_cont,var_ps] .= -(1/Red)*oneonrmat*oneonrmat
    Bstar[eq_ax,var_ws] .= -(1/Red)*oneonrmat*oneonrmat #Bstar gets an extra oneonr to account for its appearance w/o r derivs of qstar: (Bstar/r) qstar
    Bstar[eq_rad,var_ps] .= oneonrmat

    Bstar_dr[eq_cont,var_us] .= id
    Bstar_dr[eq_cont,var_ps] .= -(1/Red)*oneonrmat.*id
    Bstar_dr[eq_ax,var_ws] .= -(1/Red)*oneonrmat.*id
    Bstar_dr[eq_rad,var_ps] .= id
    #Apply Dr to right side
    Bstar_dr[eq_cont,var_us] .= Bstar_dr[eq_cont,var_us]*Dr
    Bstar_dr[eq_cont,var_ps] .= Bstar_dr[eq_cont,var_ps]*Dr
    Bstar_dr[eq_ax,var_ws] .= Bstar_dr[eq_ax,var_ws]*Dr
    Bstar_dr[eq_rad,var_ps] .= Bstar_dr[eq_rad,var_ps]*Dr
    
    Cstar[eq_cont,var_ps] .= id*(-1/Red)
    Cstar[eq_ax,var_ws] .= id*(-1/Red)
    # Cstar = Winv*Cstar*Weight

    Cstar_dr[eq_cont,var_ps] .= (-1/Red)*2*oneonrmat.*id
    Cstar_dr[eq_ax,var_ws] .= (-1/Red)*2*oneonrmat.*id
    #Apply Dr
    Cstar_dr[eq_cont,var_ps] .= Cstar_dr[eq_cont,var_ps]*Dr
    Cstar_dr[eq_ax,var_ws] .= Cstar_dr[eq_ax,var_ws]*Dr

    Cstar_drr[eq_cont,var_ps] .= id*(-1/Red)
    Cstar_drr[eq_ax,var_ws] .= id*(-1/Red)
    #Apply Drr
    Cstar_drr[eq_cont,var_ps] .= Cstar_drr[eq_cont,var_ps]*Drr
    Cstar_drr[eq_ax,var_ws] .= Cstar_drr[eq_ax,var_ws]*Drr

    Estar[eq_cont,var_ps] .= Wmat
    Estar[eq_ax,var_us] .= id
    Estar[eq_ax,var_ws] .= Wmat
    
    Brstar[eq_cont,var_ps] .= (oneonrmat.*oneonrmat)*(1/Red)
    Brstar[eq_ax,var_ws] .= (oneonrmat.*oneonrmat)*(1/Red)

    Exstar[eq_cont,var_ps] .= Wzmat
    Exstar[eq_ax,var_ws] .= Wzmat

    L = Astar .- Bstar .- Brstar .- Exstar .+ Cstar_drr .- (Bstar_dr .- Cstar_dr) .+ (1/dZ)*Estar
    
    # dirichlet conditions on u
    L[eq_cont[1],:] .= 0.0
    L[eq_cont[1],var_us[1]] = 1.0
    L[eq_cont[end],:] .= 0.0
    L[eq_cont[end],var_us[end]] = 1.0
    
    #neumann/dirichlet conditions on w
    L[eq_ax[1],:] .= 0.0
    L[eq_ax[1],var_ws] .= Dr[1,1:Nr]
    L[eq_ax[end],:] .= 0.0
    L[eq_ax[end],var_ws[end]] = 1.0

    #Condition on p
    L[eq_rad[1],:] .= 0.0
    L[eq_rad[end],:] .= 0.0
    L[eq_rad[1],var_ps[1]] = 1.0
    L[eq_rad[end],var_ps] .= Dr[end,1:Nr]
    return L
end



function b_adj(W,w_adj_prev,u_adj_prev,p_adj_prev,w,u,dZ,rvec)

    rvec[1] = 0.01
    dr = rvec[2] - rvec[1]
    k = findfirst(x -> x >= 1*(1 - 1e-3),W) #find where boundary layer ends
    k = k + 10 #Add a bit more (worked better in testing)
    bl_vel_weight = Matrix{ComplexF64}(I, Nr, Nr)
    bl_vel_weight[k:end,k:end] .= 0.0 #using parts of the free stream in the solution leads to unphysical results; here we delete those.
    bmat = zeros(ComplexF64,3*Nr)
    eq_1 = 1:Nr
    eq_2   = Nr+1:2*Nr
    eq_3  = 2*Nr+1:3*Nr
    Wmat = Diagonal(W)
    
    bmat[eq_1] .= -Wmat*p_adj_prev/dZ
    
    #Set conditions on u_adj
    bmat[eq_1[1]] = 0.0
    bmat[eq_1[end]] = 0.0

    #set conditions on w_adj
    bmat[eq_2] .= -u_adj_prev/dZ .+ -Wmat*w_adj_prev/dZ .+ bl_vel_weight*w
    bmat[eq_2[1]] = 0.0
    bmat[eq_2[end]] = 0.0

    #set conditions on p_adj
    bmat[eq_3] .= bl_vel_weight*u
    bmat[eq_3[1]] = 0.0
    bmat[eq_3[end]] = 0.0

    return bmat
end
    
    
Nr = 200
Nz = 10
w_init = zeros(Nr)
u_init = zeros(Nr)
fs_init = ones(Nr)/norm(Nz*ones(Nr))
fr_init = ones(Nr)/norm(Nz*ones(Nr))

r = complex.(r)
Dr = complex.(Dr)
Drr = complex.(Drr)
test = complex.(test)

#check to make sure functions work

lmat = L(r,Dr,Drr,complex.(test),dZ,.1,200,5000)
bmat = b(test,w_init,u_init,fs_init,fr_init,dZ,r)
lmat_adj = L_adj(r,Dr,Drr,Warray[:,100],Wz[:,100],dZ,.1,200,1000)
bmat = b_adj(Warray[:,13000], w_init, u_init,u_init,w_init,u_init, dZ, r)


# %%
Nz = 447;
z = collect(range(0,50,Nz));
dZ = z[2] - z[1];

W_sparse = Warray[:,1:30:end] #dont use full z grid, its too big, and unnecessary (original size Nz = 13400)
Wz_sparse = Wz[:,1:30:end]

uarray = zeros(ComplexF64,Nz,Nr);
warray = zeros(ComplexF64,Nz,Nr);
parray = zeros(ComplexF64,Nz,Nr);

#can choose either all ones for inital forcings or gaussian random; they both work

# fsarray = ones(ComplexF64,Nz,Nr);
# fdarray = ones(ComplexF64,Nz,Nr);

fsarray = randn(Nz,Nr);
fsarray = complex.(fsarray);
fdarray = randn(Nz,Nr);
fdarray = complex.(fdarray);
norms = norm([fsarray fdarray])

uarray_adj = zeros(ComplexF64,Nz,Nr);
warray_adj = zeros(ComplexF64,Nz,Nr);
parray_adj = zeros(ComplexF64,Nz,Nr);

function trapz2D_r_weight(F, z, r, q, W) #gives inner product of F, i.e. int int F* F r dr dz. If q, remove parts outside the b.l.
    Nz, N2r = size(F)
    Nr = div(N2r, 2)

    if q == true
        F1 = F[:, 1:Nr]
        F2 = F[:, Nr+1:end]
        for j in 1:Nz
            k = findfirst(x -> x >= 1*(1 - 1e-3),W[:,j]) #find where boundary layer ends
            k = k + 10 #Add a bit more (worked better in testing)
            bl_vel_weight = Matrix{ComplexF64}(I, Nr, Nr)
            bl_vel_weight[k:end,k:end] .= 0.0
            F1[j,:] .= bl_vel_weight*F1[j,:]
            F2[j,:] .= bl_vel_weight*F2[j,:]
        end
    else
        # Split into the two fields
        F1 = @view F[:, 1:Nr]
        F2 = @view F[:, Nr+1:end]
    end

    dz = z[2] - z[1]
    dr = r[2] - r[1]

    # trapezoid weights
    wz = ones(Nz); wz[[1, end]] .= 0.5
    wr = ones(Nr); wr[[1, end]] .= 0.5

    # 2D weights
    W = wz * wr'          # (Nz,Nr)
    R = reshape(r, 1, Nr) # (1,Nr) so it broadcasts

    # weighted |F|^2 in both components
    integrand = (conj.(F1) .* F1 + conj.(F2) .* F2)

    return sum(integrand .* R .* W) * dz * dr
end

function trapz2D_r_weight_ip(F1, F2, z, r)
    Nz, Nr = size(F1)
    @assert size(F2) == (Nz, Nr) "F1 and F2 must have same size."
    r = [r r] #double size for double fields
    dz = z[2] - z[1]
    dr = r[2] - r[1]

    # trapezoid weights
    wz = ones(Nz); wz[[1, end]] .= 0.5
    wr = ones(Nr); wr[[1, end]] .= 0.5

    # 2D weight grid
    W = wz * wr'          # (Nz, Nr)
    R = reshape(r, 1, Nr) # radial multiplier for axisymmetric measure

    integrand = conj.(F1) .* F2   # inner product pointwise

    return sum(integrand .* R .* W) * dz * dr
end

F = [fsarray fdarray]
Q = trapz2D_r_weight(F,z,real.(r),false,NaN)
fsarray = fsarray/sqrt(Q)
fdarray = fdarray/sqrt(Q) #ensures <F,F> = 1

F = [fsarray fdarray]
Q = trapz2D_r_weight(F,z,real.(r),false,NaN)

# W_sparse


Q #check if indeed <F,F> = 1

# %%

# %%
#Parameters
omega = 0
pert_norm = 10^5
gamma = 1e-1
Red = 1e5
dr = r[2] - r[1]

q_old = [copy(uarray) copy(warray)] #Keep this for iteration scheme
while abs(pert_norm) > 1e-8
    #Solve forward problem
    for j in 1:Nz- 1
        lmat = L(r,Dr,Drr,complex.(W_sparse[:,j]),dZ,omega,Nr,Red)
        bmat = b(complex.(W_sparse[:,j]), warray[j,:],uarray[j,:],fsarray[j,:],fdarray[j,:],dZ,r)
        A = sparse(lmat) \ bmat #I think sparse() doesnt do anything, but the dense solve is fast enough for this problem
        uarray[j + 1,:] .= A[1:Nr]
        warray[j + 1,:] .= A[Nr + 1:2*Nr]
        parray[j + 1,:] .= A[2*Nr + 1:3*Nr]
    end
    #Solve backward problem (adjoint)
    for j in 1:Nz - 1
        lmat = L_adj(r,Dr,Drr,W_sparse[:,end - j + 1],Wz_sparse[:,end - j + 1], dZ, omega, Nr, Red)
        bmat = b_adj(W_sparse[:, end - j + 1], warray_adj[end - j + 1,:], uarray_adj[end - j + 1,:],parray_adj[end - j + 1,:],warray[end - j + 1,:],uarray[end - j + 1,:], dZ, r)
        A = sparse(lmat) \ bmat
        uarray_adj[end - j,:] .= A[1:Nr]
        warray_adj[end - j,:] .= A[Nr + 1:2*Nr]
        parray_adj[end - j,:] .= A[2*Nr + 1:3*Nr]
    end
    #Update forcings
    fsarray = fsarray .+ gamma*warray_adj
    fdarray = fdarray .+ gamma*uarray_adj
    q = [uarray warray] #store for iteration

    #Normalize forcings so <F,F>_f = 1
    F = [fsarray fdarray]
    Q = trapz2D_r_weight(F,z,real.(r),false,NaN)
    F_true_norm = sqrt(Q)
    
    fsarray = fsarray/F_true_norm
    fdarray = fdarray/F_true_norm

    #pert norm: wait for energy of perturbations to stop changing
    pert_norm = (trapz2D_r_weight(q,z,real.(r),true,W_sparse) - trapz2D_r_weight(q_old,z,real.(r),true,W_sparse))/trapz2D_r_weight(q,z,real.(r),true,W_sparse)
    println("pert norm: $pert_norm")
    q_old = [copy(uarray) copy(warray)]
    if abs(pert_norm) > 1e-8
        uarray_adj = zeros(ComplexF64,Nz,Nr);
        warray_adj = zeros(ComplexF64,Nz,Nr);
        parray_adj = zeros(ComplexF64,Nz,Nr);
        uarray = zeros(ComplexF64,Nz,Nr);
        warray = zeros(ComplexF64,Nz,Nr);
    else
        print("completed iteration procedure")
    end
end

# %%
rflip = [-r[end:-1:1]; r]
fdarray = fdarray'
fsarray = fsarray'
magfd = sqrt.(real.(fdarray).^2 + imag.(fdarray).^2)
magfd = magfd/maximum(magfd)
fdflip = [-magfd[end:-1:1,:];magfd[:,:]]

magfs = sqrt.(real.(fsarray).^2 + imag.(fsarray).^2)
magfs = magfs/maximum(magfs)
fsflip = [magfs[end:-1:1,:];magfs[:,:]]




heatmap(real.(rflip),z,fdflip',levels=100)

# %%
heatmap(real.(rflip),z,uarray_adj,levels=100)

# %%
using LaTeXStrings
# using Plots
uarray = uarray'
warray = warray'

# magu = sqrt.(real.(uarray).^2 + imag.(uarray).^2)
magu = 2*real.(uarray)
uflip = [-magu[end:-1:1,:];magu[:,:]]

# magw = sqrt.(real.(warray).^2 + imag.(warray).^2)
magw = 2*real.(warray)
wflip = [magw[end:-1:1,:];magw[:,:]]

# magp = sqrt.(real.(parray).^2 + imag.(parray).^2)
magp = 2*real.(parray')
pflip = [magp[end:-1:1,:];magp[:,:]]

heatmap(real.(rflip),z,wflip,levels=100,xlabel=L"$r/\delta$",ylabel=L"$z/\delta$",colorbar_title=L"2Re$\{\hat{w}\}$",title=L"m = 0, $Re_δ$ = 1e5")
# savefig("m=0_red1e5_w.png")

# %%
phi = collect(range(0,2*pi,120))

magw_phi = warray[33,:]
phi_dep_magw = zeros(ComplexF64,Nz,120);
for j in 1:Nz
    for k in 1:120
        phi_dep_magw[j,k] = magw_phi[j]*exp(0im*phi[k])
    end
end

heatmap(z,phi,2*real.(phi_dep_magw)',xlabel=L"$z/δ$",ylabel=L"$\phi$",title=L"2Re$\{\hat{w}\exp{0i\phi} \}$ at $r/\delta = .8$")
# # savefig("heatmap_0m_.8.png")

# %%
r[33]

# %%
rvec = copy(r)
rvec[1] = 0.01
rmat = Diagonal(rvec)
oneonrmat = Diagonal(1 ./rvec)

rs = uarray.*conj(warray) + warray.*conj(uarray)
rsd = oneonrmat*Dr*(rmat*rs)
size(rsd)
rsd = rsd'
rsdflip = [rsd[:,end:-1:1] rsd[:,:]]
# heatmap(real.(rflip),z,real.(rsdflip),levels=100,ylabel="Coherent structures",xlabel="r")

# %%
#Find gain against ReD
#Parameters
omega = 0
# omegas = [0,1e-4,1e-3,1e-2,1e-1]
Reds = [1e2,3e2,5e2,7e2,1e3,3e3,5e3,7e3,1e4,3e4,5e4,7e4,1e5]
pert_norm = 10^5
gamma = 1e-1
# Red = 1e6
gain = zeros(size(Reds))
dr = r[2] - r[1]
l = 1
for Red in Reds
    q_old = [copy(uarray) copy(warray)] #Keep this for iteration scheme
    while abs(pert_norm) > 1e-8
        #Solve forward problem
        for j in 1:Nz- 1
            lmat = L(r,Dr,Drr,complex.(W_sparse[:,j]),dZ,omega,Nr,Red)
            bmat = b(complex.(W_sparse[:,j]), warray[j,:],uarray[j,:],fsarray[j,:],fdarray[j,:],dZ,r)
            A = sparse(lmat) \ bmat #I think sparse() doesnt do anything, but the dense solve is fast enough for this problem
            uarray[j + 1,:] .= A[1:Nr]
            warray[j + 1,:] .= A[Nr + 1:2*Nr]
            parray[j + 1,:] .= A[2*Nr + 1:3*Nr]
        end
        #Solve backward problem (adjoint)
        for j in 1:Nz - 1
            lmat = L_adj(r,Dr,Drr,W_sparse[:,end - j + 1],Wz_sparse[:,end - j + 1], dZ, omega, Nr, Red)
            bmat = b_adj(W_sparse[:, end - j + 1], warray_adj[end - j + 1,:], uarray_adj[end - j + 1,:],parray_adj[end - j + 1,:],warray[end - j + 1,:],uarray[end - j + 1,:], dZ, r)
            A = sparse(lmat) \ bmat
            uarray_adj[end - j,:] .= A[1:Nr]
            warray_adj[end - j,:] .= A[Nr + 1:2*Nr]
            parray_adj[end - j,:] .= A[2*Nr + 1:3*Nr]
        end
        #Update forcings
        fsarray = fsarray .+ gamma*warray_adj
        fdarray = fdarray .+ gamma*uarray_adj
        q = [uarray warray] #store for iteration
    
        #Normalize forcings so <F,F>_f = 1
        F = [fsarray fdarray]
        Q = trapz2D_r_weight(F,z,real.(r),false,NaN)
        F_true_norm = sqrt(Q)
        
        fsarray = fsarray/F_true_norm
        fdarray = fdarray/F_true_norm
    
        # fsarray[:,end] .= fsarray[:,end - 1]
        # fsarray[:,1] .= fsarray[:,2] #can include this to set bcs for forces but should already be covered by adjoint problem
    
        pert_norm = (trapz2D_r_weight(q,z,real.(r),true,W_sparse) - trapz2D_r_weight(q_old,z,real.(r),true,W_sparse))/trapz2D_r_weight(q,z,real.(r),true,W_sparse)
        println("pert norm: $pert_norm")
        q_old = [copy(uarray) copy(warray)]
        if abs(pert_norm) > 1e-8
            uarray_adj = zeros(ComplexF64,Nz,Nr);
            warray_adj = zeros(ComplexF64,Nz,Nr);
            parray_adj = zeros(ComplexF64,Nz,Nr);
            uarray = zeros(ComplexF64,Nz,Nr);
            warray = zeros(ComplexF64,Nz,Nr);
        else
            print("completed iteration procedure")
        end
    end
    q = [uarray warray]
    # println(l)
    # println((trapz2D_r_weight(q,z,real.(r),true,W_sparse)))
    gain[l] = (trapz2D_r_weight(q,z,real.(r),true,W_sparse))
    l += 1
    uarray = zeros(ComplexF64,Nz,Nr);
    warray = zeros(ComplexF64,Nz,Nr);
    parray = zeros(ComplexF64,Nz,Nr);
    
    #can choose either all ones for iniital forcings or gaussian random; they both work
    
    # fsarray = ones(ComplexF64,Nz,Nr);
    # fdarray = ones(ComplexF64,Nz,Nr);
    
    fsarray = randn(Nz,Nr);
    fsarray = complex.(fsarray);
    fdarray = randn(Nz,Nr);
    fdarray = complex.(fdarray);
    F = [fsarray fdarray]
    Q = trapz2D_r_weight(F,z,real.(r),false,NaN)
    fsarray = fsarray/sqrt(Q)
    fdarray = fdarray/sqrt(Q)
    
    uarray_adj = zeros(ComplexF64,Nz,Nr);
    warray_adj = zeros(ComplexF64,Nz,Nr);
    parray_adj = zeros(ComplexF64,Nz,Nr);
    pert_norm = 10^5
end

# %%
using LaTeXStrings
plot(Reds[1:l - 1],gain[1:l - 1],xscale=:log10,marker=2,xlabel=L"$Re_δ$",ylabel=(L"$\langle q, q \rangle$"),legend=false)
savefig("m=0redgain.png")

# %%
# dZ = 50/13400


#blocks for general m (works for m = 1, 2, not guranteed to be accurate for m > 2)

function L(rvec,Dr,Drr,W,dZ,omega,Nr, Red, m)
    rvec[1] = 0.01
    # Weight = Diagonal(vcat(rvec, rvec, rvec))
    # Winv   = Diagonal(vcat(1 ./ rvec, 1 ./ rvec, 1 ./ rvec))
    
    rmat = Diagonal(rvec)
    oneonrmat = Diagonal(1 ./rvec)
    # oneonrmat[1] = 0
    Wmat = Diagonal(W)
    Wrmat = Diagonal(Dr*W)
    id = Matrix{ComplexF64}(I, Nr, Nr)

    Lmat = zeros(ComplexF64, 4*Nr, 4*Nr)

    eq_cont = 1:Nr
    eq_ax   = Nr+1:2*Nr
    eq_az  = 2*Nr+1:3*Nr
    eq_rad = 3*Nr + 1:4*Nr

    var_u = 1:Nr
    var_w = Nr+1:2*Nr
    var_v = 2*Nr+1:3*Nr
    var_p = 3*Nr+1:4*Nr

    g = -1im*omega*id - (1/Red)*(-oneonrmat*oneonrmat*m^2)

    Lmat[eq_cont, var_u] .= Dr .+ oneonrmat
    Lmat[eq_cont, var_v] .= 1im*m*oneonrmat
    Lmat[eq_cont, var_w] .= 1/dZ

    

    Lmat[eq_ax, var_u] .= Wrmat
    Lmat[eq_ax, var_w] .= g .+ Wmat/dZ .- (1/Red)*(Drr .+ oneonrmat.*Dr)

    Lmat[eq_az, var_u] .= -(1/Red)*2im*m*oneonrmat*oneonrmat
    Lmat[eq_az, var_v] .= g .+ Wmat/dZ .- (1/Red)*(Drr .+ oneonrmat.*Dr .- (oneonrmat*oneonrmat))
    Lmat[eq_az, var_p] .= 1im*m*oneonrmat

    

    Lmat[eq_rad, var_u] .= g .+ Wmat/dZ .- (1/Red)*(Drr .+ oneonrmat.*Dr .- (oneonrmat*oneonrmat))
    Lmat[eq_rad, var_v] .= (1/Red)*2im*m*oneonrmat*oneonrmat
    Lmat[eq_rad, var_p] .= Dr

    #Set dirichlet conditions on u
    Lmat[1,:] .= 0

    #Symmetry conditions at r = 0 depends on if we consider m = 1 or m = 2
    if m == 1
        Lmat[1,var_u] .= Dr[1,1:Nr]
    else
        Lmat[1,var_u[1]] = 1.0 #m-dependent boundary conditions
    end
    
    Lmat[eq_cont[end],:] .= 0
    Lmat[eq_cont[end],var_u[end]] = 1 #For m ≠ 0, seems fine to set u(r=5) = 0; entrainment can happen along the disk instead
    # Lmat[eq_cont[end],var_u] .= Dr[end,1:Nr]

    #set neumann condition on w and dirichlet at edge
    Lmat[eq_ax[1],:] .= 0
    Lmat[eq_ax[1],var_w[1]] = 1.0
    Lmat[eq_ax[end],:] .= 0
    Lmat[eq_ax[end],var_w[end]] = 1

    Lmat[eq_az[1],:] .= 0
    #Symmetry conditions at r = 0 depends on if we consider m = 1 or m = 2
    if m == 1
        Lmat[eq_az[1],var_v] .= Dr[1,1:Nr]
    else
        Lmat[eq_az[1],var_v[1]] = 1.0 #m dependent boundary conditions
    end
        
    Lmat[eq_az[end],:] .= 0
    Lmat[eq_az[end],var_v[end]] = 1
    
    #set neumann/regularity condition on p
    Lmat[eq_rad[1],:] .= 0
    Lmat[eq_rad[1],var_p[1]] = 1.0
    # Lmat[eq_rad[end],var_p[1]] = rvec[1]*dr*.5
    # Lmat[eq_rad[end],var_p[end]] = rvec[end]*dr*.5 #impose ∫ p r dr  = 0 using trapezoid
    return Lmat
end

function L_adj(rvec,Dr,Drr,W,Wz,dZ,omega,Nr, Red,m)
    
    rvec[1] = 0.01
    dr = rvec[2] - rvec[1]
    Weight = dr*dZ*Diagonal(vcat(rvec, rvec, rvec))
    Winv   = Diagonal(1.0 ./ Weight)
    
    rmat = Diagonal(rvec)
    oneonrmat = Diagonal(1 ./rvec)
    Wmat = Diagonal(W)
    Wrmat = -Wmat*Dr .- oneonrmat*Wmat
    Wzmat = Diagonal(Wz)
    id = Matrix{ComplexF64}(I, Nr, Nr)
    g = -1im*omega*id - (1/Red)*(-oneonrmat*oneonrmat*m^2)
    
    Astar = zeros(ComplexF64, 4*Nr, 4*Nr)
    Bstar = zeros(ComplexF64, 4*Nr, 4*Nr)
    Cstar = zeros(ComplexF64, 4*Nr, 4*Nr)
    Cstar_drr = zeros(ComplexF64, 4*Nr, 4*Nr)
    Bstar_dr = zeros(ComplexF64, 4*Nr, 4*Nr)
    Cstar_dr = zeros(ComplexF64, 4*Nr, 4*Nr)
    
    Estar = zeros(ComplexF64, 4*Nr, 4*Nr)
    Brstar = zeros(ComplexF64, 4*Nr, 4*Nr)
    Exstar = zeros(ComplexF64, 4*Nr, 4*Nr)

    eq_cont = 1:Nr 
    eq_ax   = Nr+1:2*Nr
    eq_az  = 2*Nr+1:3*Nr
    eq_rad = 3*Nr + 1:4*Nr

    var_us = 1:Nr
    var_ws = Nr+1:2*Nr
    var_vs = 2*Nr+1:3*Nr
    var_ps = 3*Nr + 1:4*Nr

    #Adjoint operator is (A* - B*/r - ∂_r B* - ∂_x E*) + C* ∂_rr - (B*∂_r - C*∂_r) + Estar ∂_x
    
    #For Astar, just form original A and take its c.c. transpose.
    Astar[eq_cont,var_us] .= oneonrmat
    Astar[eq_cont,var_vs] .= 1im*m*oneonrmat
    Astar[eq_ax,var_us] .= Wrmat
    Astar[eq_ax,var_ws] .= g
    Astar[eq_az,var_us] .= (-1/Red)*2im*oneonrmat*oneonrmat
    Astar[eq_az,var_vs] .= g .+ (1/Red)*oneonrmat*oneonrmat
    Astar[eq_az,var_ps] .= 1im*m*oneonrmat
    Astar[eq_rad,var_us] .= g .+ (1/Red)*oneonrmat*oneonrmat
    Astar[eq_rad,var_vs] .= (1/Red)*2im*oneonrmat*oneonrmat
    Astar = Astar'
    
    Bstar[eq_cont,var_us] .= oneonrmat
    Bstar[eq_cont,var_ps] .= -(1/Red)*oneonrmat*oneonrmat
    Bstar[eq_ax,var_ws] .= -(1/Red)*oneonrmat*oneonrmat #Bstar gets an extra oneonr to account for its only appearance Bstar/r qstar
    Bstar[eq_rad,var_ps] .= oneonrmat

    Bstar_dr[eq_cont,var_us] .= id
    Bstar_dr[eq_cont,var_ps] .= -(1/Red)*oneonrmat.*id
    Bstar_dr[eq_az,var_vs] .= -(1/Red)*oneonrmat.*id
    Bstar_dr[eq_ax,var_ws] .= -(1/Red)*oneonrmat.*id
    Bstar_dr[eq_rad,var_ps] .= id

    #Apply Dr to right side
    Bstar_dr[eq_cont,var_us] .= Bstar_dr[eq_cont,var_us]*Dr
    Bstar_dr[eq_cont,var_ps] .= Bstar_dr[eq_cont,var_ps]*Dr
    Bstar_dr[eq_az,var_vs] .= Bstar_dr[eq_az,var_vs]*Dr
    Bstar_dr[eq_ax,var_ws] .= Bstar_dr[eq_ax,var_ws]*Dr
    Bstar_dr[eq_rad,var_ps] .= Bstar_dr[eq_rad,var_ps]*Dr
    
    Cstar[eq_cont,var_ps] .= id*(-1/Red)
    Cstar[eq_az,var_vs] .= id*(-1/Red)
    Cstar[eq_ax,var_ws] .= id*(-1/Red)

    Cstar_dr[eq_cont,var_ps] .= (-1/Red)*2*oneonrmat.*id
    Cstar_dr[eq_az,var_vs] .= (-1/Red)*2*oneonrmat.*id
    Cstar_dr[eq_ax,var_ws] .= (-1/Red)*2*oneonrmat.*id

    Cstar_dr[eq_cont,var_ps] .= Cstar_dr[eq_cont,var_ps]*Dr
    Cstar_dr[eq_az,var_vs] .= Cstar_dr[eq_az,var_vs]*Dr
    Cstar_dr[eq_ax,var_ws] .= Cstar_dr[eq_ax,var_ws]*Dr

    Cstar_drr[eq_cont,var_ps] .= id*(-1/Red)
    Cstar_drr[eq_az,var_vs] .= id*(-1/Red)
    Cstar_drr[eq_ax,var_ws] .= id*(-1/Red)

    Cstar_drr[eq_cont,var_ps] .= Cstar_drr[eq_cont,var_ps]*Drr
    Cstar_drr[eq_az,var_vs] .= Cstar_drr[eq_az,var_vs]*Drr
    Cstar_drr[eq_ax,var_ws] .= Cstar_drr[eq_ax,var_ws]*Drr

    Estar[eq_cont,var_ps] .= Wmat
    Estar[eq_ax,var_us] .= id
    Estar[eq_ax,var_ws] .= Wmat
    Estar[eq_az,var_vs] .= Wmat


    Brstar[eq_cont,var_ps] .= (oneonrmat.*oneonrmat)*(1/Red)
    Brstar[eq_az,var_vs] .= (oneonrmat.*oneonrmat)*(1/Red)
    Brstar[eq_ax,var_ws] .= (oneonrmat.*oneonrmat)*(1/Red)


    Exstar[eq_cont,var_ps] .= -Wzmat
    Exstar[eq_ax,var_ws] .= -Wzmat
    Exstar[eq_az,var_vs] .= -Wzmat

    L = Astar .- Bstar .- Brstar .- Exstar .+ Cstar_drr .- (Bstar_dr .- Cstar_dr) .+ (1/dZ)*Estar
    
    # dirichlet conditions on u
    L[eq_cont[1],:] .= 0.0
    
    # Symmetry conditions at r = 0 depend on m
    if m == 1
        L[eq_cont[1],var_us] .= Dr[1,1:Nr]
    else
        L[eq_cont[1],var_us[1]] = 1.0
    end
    L[eq_cont[end],:] .= 0.0
    L[eq_cont[end],var_us[end]] = 1.0

    #v conditions

    L[eq_az[1],:] .= 0.0
    # Symmetry conditions at r = 0 depend on m
    if m == 1
        L[eq_az[1],var_vs] .= Dr[1,1:Nr]
    else
        L[eq_az[1],var_vs[1]] = 1.0
    end
    L[eq_az[end],:] .= 0.0
    L[eq_az[end],var_vs[end]] = 1.0
    
    #dirichlet conditions on w
    L[eq_az[end],:] .= 0.0
    L[eq_az[end],var_vs[end]] = 1.0

    #Neumann condition on p
    L[eq_rad[1],:] .= 0.0
    L[eq_rad[1],var_ps[1]] = 1.0
    return L
end
    




function b(W,wprev,vprev,uprev,fs,fa,fd,dZ,rvec)

    bmat = zeros(ComplexF64,4*Nr)
    Wmat = Diagonal(W)
    eq_cont = 1:Nr
    eq_ax   = Nr+1:2*Nr
    eq_az  = 2*Nr+1:3*Nr
    eq_rad = 3*Nr + 1:4*Nr
    
    bmat[eq_cont] .= wprev/dZ
    # radial bcs
    bmat[1] = 0.0
    bmat[eq_cont[end]] = 0.0

   
    bmat[eq_ax] .= Wmat*wprev/dZ .+ fs
    #axial bcs
    bmat[eq_ax[1]] = 0.0
    bmat[eq_ax[end]] = 0.0

    bmat[eq_az] .= Wmat*vprev/dZ .+ fa
    #pressure regularization
    bmat[eq_az[1]] = 0.0
    bmat[eq_az[end]] = 0.0
    
    bmat[eq_rad] .= Wmat*uprev/dZ .+ fd
    
    #pressure regularization/boundary condition
    bmat[eq_rad[1]] = 0.0

    return bmat
end



function b_adj(W,w_adj_prev,v_adj_prev,u_adj_prev,p_adj_prev,w,v,u,dZ,rvec)

    rvec[1] = 0.01
    dr = rvec[2] - rvec[1]
    
    k = findfirst(x -> x >= 1*(1 - 1e-3),W) #find where boundary layer ends
    k = k + 10 #Add a bit more (worked better in testing)
    bl_vel_weight = Matrix{ComplexF64}(I, Nr, Nr)
    bl_vel_weight[k:end,k:end] .= 0.0
    
    bmat = zeros(ComplexF64,4*Nr)
    eq_1 = 1:Nr
    eq_2   = Nr+1:2*Nr
    eq_3  = 2*Nr+1:3*Nr
    eq_4 = 3*Nr + 1:4*Nr
    Wmat = Diagonal(W)
    
    bmat[eq_1] .= -Wmat*v_adj_prev/dZ
    bmat[eq_1[1]] = 0.0
    bmat[eq_1[end]] = 0.0

    
    bmat[eq_2] .= -u_adj_prev/dZ .+ -Wmat*w_adj_prev/dZ .+ bl_vel_weight*w
    bmat[eq_2[1]] = 0.0
    bmat[eq_2[end]] = 0.0

    bmat[eq_3] .= -Wmat*v_adj_prev/dZ .+ bl_vel_weight*v
    bmat[eq_3[1]] = 0.0
    bmat[eq_3[end]] = 0.0
    
    bmat[eq_4] .= bl_vel_weight*u
    bmat[eq_4[1]] = 0.0

    return bmat
end
    
    
Nr = 200
Nz = 10
w_init = zeros(Nr)
v_init = zeros(Nr)
u_init = zeros(Nr)
fs_init = ones(Nr)/norm(Nz*ones(Nr))
fa_init = ones(Nr)/norm(Nz*ones(Nr))
fr_init = ones(Nr)/norm(Nz*ones(Nr))

r = complex.(r)
Dr = complex.(Dr)
Drr = complex.(Drr)
test = complex.(test)

#check to make sure functions work

# lmat = L(r,Dr,Drr,complex.(test),dZ,.1,200,5000,1)
# bmat = b(test,w_init,v_init,u_init,fs_init,fa_init,fr_init,dZ,r)
# lmat_adj = L_adj(r,Dr,Drr,Warray[:,100],Wz[:,100],dZ,.1,200,1000,1)
# bmat = b_adj(Warray[:,13000], w_init, v_init,u_init,u_init,w_init,u_init, dZ, r)

# print(cond(lmat))

# plot(real.(r),real.(Warray[:,13000]))


# %%
"""
Compute axisymmetric kinetic-energy inner product:
 ∫∫ (u1*u2 + v1*v2 + w1*w2) * r dr dz
Apply boundary-layer masking when q=true using W(r,z).

All fields must be (Nr, Nz).
"""
function trapz2D_axisym_ip_masked(u1, v1, w1,
                                  u2, v2, w2,
                                  r, z, W; q=false)
    Nr, Nz = size(u1)
    @assert size(v1) == (Nr, Nz)
    @assert size(w1) == (Nr, Nz)
    @assert size(u2) == (Nr, Nz)
    @assert size(v2) == (Nr, Nz)
    @assert size(w2) == (Nr, Nz)
    @assert size(W)  == (Nr, Nz)

    # -------------------------------
    # Apply boundary-layer masking
    # -------------------------------
    if q
        for j in 1:Nz
            k = findfirst(x -> x >= (1 - 1e-3), W[:, j])
            k = (k === nothing) ? Nr : min(k + 10, Nr)

            # Zero outside boundary layer for all components
            u1[k:Nr, j] .= 0
            v1[k:Nr, j] .= 0
            w1[k:Nr, j] .= 0

            u2[k:Nr, j] .= 0
            v2[k:Nr, j] .= 0
            w2[k:Nr, j] .= 0
        end
    end

    # -------------------------------------
    # Axisymmetric kinetic energy integrand
    # -------------------------------------
    integrand =
        conj.(u1).*u2 .+
        conj.(v1).*v2 .+
        conj.(w1).*w2        # (Nr, Nz)

    # trapezoid weights
    wr = ones(Nr); wr[[1,end]] .= 0.5
    wz = ones(Nz); wz[[1,end]] .= 0.5

    # axisymmetric weight r * wr * wz
    R = r .* wr              # Nr
    W2D = R * wz'            # Nr × Nz

    dr = r[2] - r[1]
    dz = z[2] - z[1]

    return sum(integrand .* W2D) * dr * dz
end



Nz = 447;
z = collect(range(0,50,Nz));
dZ = z[2] - z[1];

W_sparse = Warray[:,1:30:end]
Wz_sparse = Wz[:,1:30:end]

uarray = zeros(ComplexF64,Nr,Nz);
varray = zeros(ComplexF64,Nr,Nz);
warray = zeros(ComplexF64,Nr,Nz);
parray = zeros(ComplexF64,Nr,Nz);

#can choose either all ones for iniital forcings or gaussian random; they both work

# fsarray = ones(ComplexF64,Nz,Nr);
# fdarray = ones(ComplexF64,Nz,Nr);

fsarray = randn(Nr,Nz);
fsarray = complex.(fsarray);
faarray = randn(Nr,Nz);
faarray = complex.(faarray);
fdarray = randn(Nr,Nz);
fdarray = complex.(fdarray);

uarray_adj = zeros(ComplexF64,Nr,Nz);
varray_adj = zeros(ComplexF64,Nr,Nz);
warray_adj = zeros(ComplexF64,Nr,Nz);
parray_adj = zeros(ComplexF64,Nr,Nz);

Q = trapz2D_axisym_ip_masked(fsarray,faarray,fdarray,fsarray,faarray,fdarray,real.(r),z,W_sparse)
fsarray = fsarray/Q^(1/2)
faarray = faarray/Q^(1/2)
fdarray = fdarray/Q^(1/2)
Q = trapz2D_axisym_ip_masked(fsarray,faarray,fdarray,fsarray,faarray,fdarray,real.(r),z,W_sparse)

Q
# W_sparse

# %%
#Parameters
omega = 0
pert_norm = 10^5
gamma = 1e-1
Red = 1e3
m = 1
dr = r[2] - r[1]

uarray_old = copy(uarray)
varray_old = copy(varray)
warray_old = copy(warray)

while abs(pert_norm) > 1e-8
    #Solve forward problem
    for j in 1:Nz- 1
        lmat = L(r,Dr,Drr,complex.(W_sparse[:,j]),dZ,omega,Nr,Red,m)
        bmat = b(complex.(W_sparse[:,j]), warray[:,j],varray[:,j],uarray[:,j],fsarray[:,j],faarray[:,j],fdarray[:,j],dZ,r)
        A = sparse(lmat) \ bmat #I think sparse() doesnt do anything, but the dense solve is fast enough for this problem
        uarray[:,j + 1] .= A[1:Nr]
        warray[:,j + 1] .= A[Nr + 1:2*Nr]
        varray[:,j + 1] .= A[2*Nr + 1:3*Nr]
        parray[:,j + 1] .= A[3*Nr + 1:4*Nr]
    end
    #Solve backward problem (adjoint)
    for j in 1:Nz - 1
        lmat = L_adj(r,Dr,Drr,W_sparse[:,end - j + 1],Wz_sparse[:,end - j + 1], dZ, omega, Nr, Red,m)
        bmat = b_adj(W_sparse[:, end - j + 1], warray_adj[:,end - j + 1],varray_adj[:,end - j + 1], uarray_adj[:,end - j + 1],parray_adj[:,end - j + 1],warray[:,end - j + 1],varray[:,end - j + 1],uarray[:,end - j + 1], dZ, r)
        A = sparse(lmat) \ bmat
        uarray_adj[:,end - j] .= A[1:Nr]
        warray_adj[:,end - j] .= A[Nr + 1:2*Nr]
        varray_adj[:,end - j] .= A[2*Nr + 1:3*Nr]
        parray_adj[:,end - j] .= A[3*Nr + 1:4*Nr]
    end
    #Update forcings
    fsarray = fsarray .+ gamma*warray_adj
    faarray = faarray .+ gamma*varray_adj
    fdarray = fdarray .+ gamma*uarray_adj
    q = [uarray varray warray] #store for iteration

    #Normalize forcings so <F,F>_f = 1
    Q = trapz2D_axisym_ip_masked(fsarray,faarray,fdarray,fsarray,faarray,fdarray,real.(r),z,W_sparse)
    fsarray = fsarray/Q^(1/2)
    faarray = faarray/Q^(1/2)
    fdarray = fdarray/Q^(1/2)
    
    # fsarray = fsarray/F_true_norm
    # fdarray = fdarray/F_true_norm

    # fsarray[:,end] .= fsarray[:,end - 1]
    # fsarray[:,1] .= fsarray[:,2] #can include this to set bcs for forces but should already be covered by adjoint problem

    pert_norm = (trapz2D_axisym_ip_masked(uarray,varray,warray,uarray,varray,warray,real.(r),z,W_sparse,q=true) - trapz2D_axisym_ip_masked(uarray_old,varray_old,warray_old,uarray_old,varray_old,warray_old,real.(r),z,W_sparse,q=true))/trapz2D_axisym_ip_masked(uarray,varray,warray,uarray,varray,warray,real.(r),z,W_sparse,q=true)
    println("pert norm: $pert_norm")
    uarray_old = copy(uarray)
    varray_old = copy(varray)
    warray_old = copy(warray)
    if abs(pert_norm) > 1e-8
        uarray_adj = zeros(ComplexF64,Nr,Nz);
        varray_adj = zeros(ComplexF64,Nr,Nz);
        warray_adj = zeros(ComplexF64,Nr,Nz);
        parray_adj = zeros(ComplexF64,Nr,Nz);
        uarray = zeros(ComplexF64,Nr,Nz);
        varray = zeros(ComplexF64,Nr,Nz);
        warray = zeros(ComplexF64,Nr,Nz);
    else
        print("completed iteration procedure")
    end
end

# %%
rflip = [-r[end:-1:1]; r]
magfd = sqrt.(real.(fdarray).^2 + imag.(fdarray).^2)
magfd = magfd/maximum(magfd)
fdflip = [-magfd[end:-1:1,:];magfd[:,:]]

magfs = sqrt.(real.(fsarray).^2 + imag.(fsarray).^2)
magfs = magfs/maximum(magfs)
fsflip = [-magfs[end:-1:1,:];magfs[:,:]]

magfa = sqrt.(real.(faarray).^2 + imag.(faarray).^2)
magfa = magfa/maximum(magfa)
faflip = [-magfa[end:-1:1,:];magfa[:,:]]


# heatmap(real.(rflip),z,fdflip',levels=100)

# %%
# heatmap(real.(rflip),z,fsflip',levels=100)

# %%
# heatmap(real.(rflip),z,faflip',levels=100)

# %%
using LaTeXStrings
pyplot()
# using Plots
uarray = uarray'
warray = warray'

magu = 2*real.(uarray)
uflip = [magu[end:-1:1,:];magu[:,:]]

magw = 2*real.(warray)
wflip = [-magw[end:-1:1,:];magw[:,:]]

magp = 2*real.(parray')
pflip = [-magp[end:-1:1,:];magp[:,:]]

# heatmap(real.(rflip),z,wflip',levels=100,xlabel=L"r",ylabel=L"2Re$\{\hat{w}\}$",title=L"m = 1, $Re_δ$ = 1e5")
heatmap(real.(rflip),z,wflip',levels=100,xlabel=L"$r/\delta$",ylabel=L"$z/\delta$",colorbar_title=L"2Re$\{\hat{w}\}$",title=L"m = 2, $Re_δ$ = 1e5")
# savefig("m=2_red1e5_w.png")

# %%
# heatmap(real.(rflip),z,uflip',levels=100)

# %%
# heatmap(real.(rflip),z,pflip',levels=100)

# %%

# %%
phi = collect(range(0,2*pi,120))

magw_phi = warray[15,:]
phi_dep_magw = zeros(ComplexF64,Nz,120);
for j in 1:Nz
    for k in 1:120
        phi_dep_magw[j,k] = magw_phi[j]*exp(1im*phi[k])
    end
end

heatmap(z,phi,2*real.(phi_dep_magw)',xlabel=L"$z/δ$",ylabel=L"$\phi$",title=L"2Re$\{\hat{w}\exp{1i\phi} \}$ at $r/\delta = .35$")
savefig("heatmap_1m_.35.png")

# %%
r[15]

# %%
rvec = copy(r)
rvec[1] = 0.01
rmat = Diagonal(rvec)
oneonrmat = Diagonal(1 ./rvec)

rs = uarray.*conj(warray) + warray.*conj(uarray)
rsd = oneonrmat*Dr*(rmat*rs)
rsd = rsd'
rsd = rsd/norm(rsd)
rsdflip = -[rsd[:,end:-1:1] rsd[:,:]]
heatmap(real.(rflip),z,real.(rsdflip),levels=100,ylabel=L"RSD = -$\frac{1}{r} ∂_r (r u^* w + w^* u)$",xlabel="r")

# %%
# plot(real.(rflip),real.(rsdflip[2,:]))

# %%
#Parameters

#Plot gain v.s. Red (takes a while... get a cup of coffee)

omega = 0
pert_norm = 10^5
gamma = 1e-1
Reds = [1e2,3e2,5e2,7e2,1e3,3e3,5e3,7e3,1e4,3e4,5e4,7e4,1e5]
gain = zeros(size(Reds))
m = 1
l = 1
dr = r[2] - r[1]

# q_old = [copy(uarray) copy(varray) copy(warray)] #Keep this for iteration scheme
uarray_old = copy(uarray)
varray_old = copy(varray)
warray_old = copy(warray)

for Red in Reds
    # l = l + 1
    while abs(pert_norm) > 1e-8
        #Solve forward problem
        for j in 1:Nz- 1
            lmat = L(r,Dr,Drr,complex.(W_sparse[:,j]),dZ,omega,Nr,Red,m)
            bmat = b(complex.(W_sparse[:,j]), warray[:,j],varray[:,j],uarray[:,j],fsarray[:,j],faarray[:,j],fdarray[:,j],dZ,r)
            A = sparse(lmat) \ bmat #I think sparse() doesnt do anything, but the dense solve is fast enough for this problem
            uarray[:,j + 1] .= A[1:Nr]
            warray[:,j + 1] .= A[Nr + 1:2*Nr]
            varray[:,j + 1] .= A[2*Nr + 1:3*Nr]
            parray[:,j + 1] .= A[3*Nr + 1:4*Nr]
        end
        #Solve backward problem (adjoint)
        for j in 1:Nz - 1
            lmat = L_adj(r,Dr,Drr,W_sparse[:,end - j + 1],Wz_sparse[:,end - j + 1], dZ, omega, Nr, Red,m)
            bmat = b_adj(W_sparse[:, end - j + 1], warray_adj[:,end - j + 1],varray_adj[:,end - j + 1], uarray_adj[:,end - j + 1],parray_adj[:,end - j + 1],warray[:,end - j + 1],varray[:,end - j + 1],uarray[:,end - j + 1], dZ, r)
            A = sparse(lmat) \ bmat
            uarray_adj[:,end - j] .= A[1:Nr]
            warray_adj[:,end - j] .= A[Nr + 1:2*Nr]
            varray_adj[:,end - j] .= A[2*Nr + 1:3*Nr]
            parray_adj[:,end - j] .= A[3*Nr + 1:4*Nr]
        end
        #Update forcings
        fsarray = fsarray .+ gamma*warray_adj
        faarray = faarray .+ gamma*varray_adj
        fdarray = fdarray .+ gamma*uarray_adj
        q = [uarray varray warray] #store for iteration
    
        #Normalize forcings so <F,F>_f = 1
        Q = trapz2D_axisym_ip_masked(fsarray,faarray,fdarray,fsarray,faarray,fdarray,real.(r),z,W_sparse)
        fsarray = fsarray/Q^(1/2)
        faarray = faarray/Q^(1/2)
        fdarray = fdarray/Q^(1/2)
        
        # fsarray = fsarray/F_true_norm
        # fdarray = fdarray/F_true_norm
    
        # fsarray[:,end] .= fsarray[:,end - 1]
        # fsarray[:,1] .= fsarray[:,2] #can include this to set bcs for forces but should already be covered by adjoint problem
    
        pert_norm = (trapz2D_axisym_ip_masked(uarray,varray,warray,uarray,varray,warray,real.(r),z,W_sparse,q=true) - trapz2D_axisym_ip_masked(uarray_old,varray_old,warray_old,uarray_old,varray_old,warray_old,real.(r),z,W_sparse,q=true))/trapz2D_axisym_ip_masked(uarray,varray,warray,uarray,varray,warray,real.(r),z,W_sparse,q=true)
        println("pert norm: $pert_norm")
        # q_old = [copy(uarray) copy(warray)]
        uarray_old = copy(uarray)
        varray_old = copy(varray)
        warray_old = copy(warray)
        if abs(pert_norm) > 1e-8
            uarray_adj = zeros(ComplexF64,Nr,Nz);
            varray_adj = zeros(ComplexF64,Nr,Nz);
            warray_adj = zeros(ComplexF64,Nr,Nz);
            parray_adj = zeros(ComplexF64,Nr,Nz);
            uarray = zeros(ComplexF64,Nr,Nz);
            varray = zeros(ComplexF64,Nr,Nz);
            warray = zeros(ComplexF64,Nr,Nz);
        else
            print("completed iteration procedure")
        end
    end
    q = [uarray varray warray]
    # println(l)
    # println((trapz2D_r_weight(q,z,real.(r),true,W_sparse)))
    gain[l] = trapz2D_axisym_ip_masked(uarray,varray,warray,uarray,varray,warray,real.(r),z,W_sparse,q=true)
    l += 1
    uarray = zeros(ComplexF64,Nr,Nz);
    warray = zeros(ComplexF64,Nr,Nz);
    parray = zeros(ComplexF64,Nr,Nz);
    
    fsarray = randn(Nr,Nz);
    fsarray = complex.(fsarray);
    faarray = randn(Nr,Nz);
    faarray = complex.(faarray);
    fdarray = randn(Nr,Nz);
    fdarray = complex.(fdarray);
    # norms = norm([fsarray faarray fdarray])
    
    uarray_adj = zeros(ComplexF64,Nr,Nz);
    varray_adj = zeros(ComplexF64,Nr,Nz);
    warray_adj = zeros(ComplexF64,Nr,Nz);
    parray_adj = zeros(ComplexF64,Nr,Nz);
    # trapz2D_axisym_ip_masked(F1, F2, r, z, W; q=false)
    # F = [fsarray faarray fdarray]
    Q = trapz2D_axisym_ip_masked(fsarray,faarray,fdarray,fsarray,faarray,fdarray,real.(r),z,W_sparse)
    fsarray = fsarray/Q^(1/2)
    faarray = faarray/Q^(1/2)
    fdarray = fdarray/Q^(1/2)
    pert_norm = 10^5
end

# %%
#Plot centerline velocity

plot(z,W_sparse[1,:],label=false,xlabel="z/δ",ylabel=L"$W(r=0,z)$",title="Centerline velocity of W")
savefig("Cent_line_vel.png")

# %%
using Kronecker

function L_2d(rvec,Dr,Drr,W,Dz,omega,Nr,Nz,Red, m)
    rvec[1] = 0.01
    # Weight = Diagonal(vcat(rvec, rvec, rvec))
    # Winv   = Diagonal(vcat(1 ./ rvec, 1 ./ rvec, 1 ./ rvec))
    N = Nr*Nz
    rmat = Diagonal(vcat(rvec, rvec, rvec))
    oneonrmat = Diagonal(vcat(1 ./rvec,1 ./rvec,1 ./rvec))
    Wmat = Diagonal(vcat(W,W,W))
    Wrmat = Diagonal(vcat(Dr*W,Dr*W,Dr*W))
    id_z = Matrix{ComplexF64}(I, Nz, Nz)
    id_r = Matrix{ComplexF64}(I, Nr, Nr)
    id = Matrix{ComplexF64}(I,N,N)

    Lmat = zeros(ComplexF64, 4*N, 4*N)

    eq_cont = 1:N
    eq_ax   = N+1:2*N
    eq_az  = 2*N+1:3*N
    eq_rad = 3*N + 1:4*N

    var_u = 1:N
    var_w = N+1:2*N
    var_v = 2*N+1:3*N
    var_p = 3*N+1:4*N

    g = -1im*omega*id - (1/Red)*(-oneonrmat*oneonrmat*m^2)

    # Lmat[eq_cont, var_u] .= Dr .+ oneonrmat
    Lmat[eq_cont, var_u] .= kron(id_z,Dr) .+ oneonrmat
    Lmat[eq_cont, var_v] .= 1im*m*oneonrmat
    Lmat[eq_cont, var_w] .= kron(Dz,id_r)

    

    Lmat[eq_ax, var_u] .= Wrmat
    Lmat[eq_ax, var_w] .= g .+ Wmat.*kron(Dz,id_r) .- (1/Red)*(kron(id_z,Drr) .+ oneonrmat.*kron(id_z,Dr))

    Lmat[eq_az, var_u] .= -(1/Red)*2im*m*oneonrmat.*oneonrmat
    Lmat[eq_az, var_v] .= g .+ Wmat.*kron(Dz,id_r) .- (1/Red)*(kron(id_z,Drr) .+ oneonrmat.*kron(id_z,Dr) .- (oneonrmat*oneonrmat))
    Lmat[eq_az, var_p] .= 1im*m*oneonrmat

    

    Lmat[eq_rad, var_u] .= g .+ Wmat.*kron(Dz,id_r) .- (1/Red)*(kron(id_z,Drr) .+ oneonrmat.*kron(id_z,Dr) .- (oneonrmat*oneonrmat))
    Lmat[eq_rad, var_v] .= (1/Red)*2im*m*oneonrmat*oneonrmat
    Lmat[eq_rad, var_p] .= kron(id_z,Dr)

    #Set dirichlet conditions on u
    Lmat[1,:] .= 0

    #Symmetry conditions at r = 0 depends on if we consider m = 1 or m = 2
    if m == 1
        Lmat[1,var_u] .= Dr[1,1:Nr]
    else
        Lmat[1,var_u[1]] = 1.0 #m-dependent boundary conditions
    end
    
    Lmat[eq_cont[end],:] .= 0
    Lmat[eq_cont[end],var_u[end]] = 1 #For m ≠ 0, seems fine to set u(r=5) = 0; entrainment can happen along the disk instead
    # Lmat[eq_cont[end],var_u] .= Dr[end,1:Nr]

    #set neumann condition on w and dirichlet at edge
    Lmat[eq_ax[1],:] .= 0
    Lmat[eq_ax[1],var_w[1]] = 1.0
    Lmat[eq_ax[end],:] .= 0
    Lmat[eq_ax[end],var_w[end]] = 1

    Lmat[eq_az[1],:] .= 0
    #Symmetry conditions at r = 0 depends on if we consider m = 1 or m = 2
    if m == 1
        Lmat[eq_az[1],var_v] .= Dr[1,1:Nr]
    else
        Lmat[eq_az[1],var_v[1]] = 1.0 #m dependent boundary conditions
    end
        
    Lmat[eq_az[end],:] .= 0
    Lmat[eq_az[end],var_v[end]] = 1
    
    #set neumann/regularity condition on p
    Lmat[eq_rad[1],:] .= 0
    Lmat[eq_rad[1],var_p[1]] = 1.0
    # Lmat[eq_rad[end],var_p[1]] = rvec[1]*dr*.5
    # Lmat[eq_rad[end],var_p[end]] = rvec[end]*dr*.5 #impose ∫ p r dr  = 0 using trapezoid
    return Lmat
end

# %%

# %%

# %%
