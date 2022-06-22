### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 875e4f2c-80d6-11eb-00e8-5362515a5d82
using VoronoiFVM, ForwardDiff, Plots, Markdown, LaTeXStrings, ExtendableGrids, PlutoUI, GridVisualize, PyPlot, Printf, ElasticArrays,StaticArrays

# ╔═╡ e7d187f2-ee2b-4e7c-bf33-3d4351beac3f
md"""
This is a supplementary notebook for 
- Petr Vágner, Michal Pavelka, Jürgen Fuhrmann, Václav Klika. *A multiscale thermodynamic generalization of the Maxwell-Stefan diffusion equations and of the dusty gas model*, submitted to Intl J Heat and Mass Transfer on (??-06-2020).

"""

# ╔═╡ a4c5960a-ece3-4f86-a761-c7f8169c7a7f
PlutoUI.TableOfContents(title="Non-isothermal dusty gas model")

# ╔═╡ 694fcd71-4691-45bb-a213-e2d83a19adf2
md"""
# Governing system
"""

# ╔═╡ adaa6907-1895-4ed0-af29-fbeb2302f8c0
md"""
#### Mass and energy balances
"""

# ╔═╡ fc63abe0-8285-11eb-11d3-43457bd58690
md"""
```math
\begin{align}
\partial_t n_a + \partial_i J^i_a &= 0
~,
\\
\partial_t n_b + \partial_i J^i_b &= 0
~,
\\
\partial_t \left(
	T\left[
		c^V_D n_D  + c^V_a n_a  + c^V_b n_b 
	\right]
\right)
+\partial_i\left(
	-\lambda\partial_i T
	+T\left[
 		c^V_a\gamma_aJ^i_a
		+c^V_b\gamma_bJ^i_b
  \right]
\right)
&= 0
~.
\end{align}
```
"""

# ╔═╡ 2a8beb1f-c83b-464a-8619-c3a8fd05b9bf
md"""
##### Force-flux relations
"""

# ╔═╡ 78e067ea-0897-40e4-9c23-85272b7bc1d8
md"""
```math
\begin{align}
    F^i_{a}:=
      \partial_i n_a + n_a \partial_i \ln T
    &=
    \frac{1 }{n}
    \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}\left(
        n_a J^i_b - n_b J^i_a
    \right)
    -
    \frac{1}{\frac{\varepsilon}{\tau}D_{aD}}
        J^i_a
    \\
    &=
    \frac{n_a}{n}
    \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}
        J^i_b 
    -
    \left( 
		  \frac{1}{n}
          \frac{n_b }{\frac{\varepsilon}{\tau}D_{ab}}
          + \frac{1}{\frac{\varepsilon}{\tau}D_{aD}}
    \right)
    J^i_a
    ~,
    \\
    F^i_{b}:=  
    \partial_i n_b +n_b  \partial_i \ln T
    &=
    \frac{n_b}{n}
    \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}
        J^i_a 
    -
    \left(
        \frac{1}{\frac{\varepsilon}{\tau}D_{bD}} 
		+ \frac{1}{n}\frac{n_a}{\frac{\varepsilon}{\tau}D_{ab}}
    \right)
         J^i_b
    ~.
\end{align}
```
"""

# ╔═╡ 59e1d95e-9508-442c-b554-1dc448c41549
md"""
``n=n_a+n_b`` denotes the total number density 
"""

# ╔═╡ ab654b48-80e8-11eb-20af-dd0d3facedc8
md"""
Using vector notation, the force-flux relations read
```math
\begin{align} 
\begin{bmatrix}
F^i_{a} \\
F^i_{b} \\
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
    -
    \left( 
		\frac{n_b}{n}
          \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}
          + \frac{1}{\frac{\varepsilon}{\tau}D_{aD}}
    \right)
&
    \frac{n_a}{n}
    \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}  \\
 \\
 \frac{n_b}{n}
 \frac{1}{\frac{\varepsilon}{\tau}D_{ab}}
&
    -
    \left(
        \frac{1}{\frac{\varepsilon}{\tau}D_{bD}} 
		+ \frac{n_a}{n}\frac{1}{\frac{\varepsilon}{\tau}D_{ab}}
    \right)
\\
\end{bmatrix}
}_{
\mathbf{M}
}
\begin{bmatrix}
J^i_{a} \\
J^i_{b} \\
\end{bmatrix}
\end{align}
```
and thus the matrix ``\mathbf{M}`` is defined.
"""

# ╔═╡ dc840c0c-ac34-44da-9eac-fb2897665b6f
md"""
# Finite volume method
We use the two-point flux Voronoi box cell method
"""

# ╔═╡ 0d68b27a-1d5f-4a01-ad9f-0e894b07c508
md"""
#### Discrete temperature and densities
For a control volume $\omega_k$ associated with the collocation point $\mathbf{x}_k\in\omega_k$, the discrete values $T_k, n_{a,k}$ and $n_{a,k}$ as follows:
```math
\begin{align}
    n_{\alpha, k} |\omega_k|
	&:=
    \int_{\omega_k} n_\alpha \mathrm{d}\mathbf{x}
	~,
	\quad\alpha\in\{a,b\}
    ~,
    \\
	T_{k} |\omega_k| 
    &:=
    \int_{\omega_k} T   \mathrm{d}\mathbf{x}
	~.
\end{align}
```
"""

# ╔═╡ b667d0b2-57e1-4de3-9197-776720d801cc
md"""
## Discrete balances
```math
\begin{align}
|\omega_k|\partial_t n_{\alpha, k}
+\sum_{\sigma_{kl}\neq\emptyset}|\sigma_{kl}| J^{\textrm{num}}_{\alpha, kl} =0
\end{align}
```
```math
\begin{align}
|\omega_k| \partial_t 
\left(
    T_{k} \left[
			c^V_a n_{a,k}
		   +c^V_b n_{b,k}
		   +c^V_D n_{D}
	\right]
\right)
+\sum_{\sigma_{kl}\neq\emptyset}|\sigma_{kl}| \left(     
    J^{\textrm{num}}_{T, kl}
\right)
=
0
~,
\end{align}
```
where $J^{\textrm{num}}_{\alpha, kl}$ is the numerical flux across the edge $\sigma_{kl} = \omega_k\cap\omega_k$.
"""

# ╔═╡ 5389b0b8-dfed-4f3d-bdff-943356fd147a
md"""
##### Discrete heat flux
```math
\begin{align}
J^{\textrm{num}}_{T, kl} 
= 
\frac{\lambda}{h_kl}\left[ 
	T_{l} B(Q_{T,kl})
   -T_{k} B(-Q_{T,kl})
\right]
\\
\text{where}\quad Q_{T,kl}  
= 
\frac{1}{h_{kl}\lambda}
\sum_{\alpha=a,b} \gamma_\alpha c^V_\alpha J^{\textrm{num}}_{\alpha, kl}
\end{align}
```
"""

# ╔═╡ 3a9587bd-0beb-49eb-a074-edd7e4f89924
md"""
#### Discrete force-flux relation
"""

# ╔═╡ 5c2a32be-8179-11eb-3aa1-e569dc27c378
md"""
```math
\begin{align}
    F^{\textrm{num}}_{\alpha, kl} 
	&= 
	\frac{1}{h_{kl}}
	\frac{c_\alpha^V}{k_B}\left( \gamma_\alpha - 1 \right)
  	\left[ 
       n_{\alpha, l} B(Q_{kl})
      -n_{\alpha, k} B(-Q_{kl})
  	\right]
  ~,
  \\
  Q_{kl}  &= -\frac{1}{h_{kl}}(\log T_l - \log T_k)
\end{align}
```
Bernoulli function
``B(x) = \frac{x}{e^x-1}``, distance of collocation points
``h_{kl} = |\mathbf{x}_k - \mathbf{x}_l|``
"""

# ╔═╡ ce4c6012-8280-11eb-3260-714f867c2c67
md"""

#### Discrete $\mathbf{M}$
```math
\begin{align}
M^{\textrm{num}}_{kl}
=
\begin{bmatrix}
    -
    \left( 
          \frac{\bar{n}_{b,kl} }{\bar{n}_{kl}}\frac{1}{D_{ab}}
          + \frac{1}{D_{aD}}
    \right)
&
    \frac{\bar{n}_{a,kl}}{\bar{n}_{kl}}
    \frac{1}{D_{ab}}  \\
 \\
 	\frac{\bar{n}_{b,kl}}{\bar{n}_{kl}}
 	\frac{1}{D_{ab}}
&
    -
    
    \left(
        \frac{\bar{n}_{a,kl}}{\bar{n}_{kl}}\frac{1}{D_{ab}}
		+\frac{1}{D_{bD}} 
    \right)
\\
\end{bmatrix}
\end{align}
```
"""

# ╔═╡ f91b77a7-66ec-40e7-aaef-79aad206a45f
md"""
where
``\bar{n}_{\alpha,kl} := \tfrac{1}{2}\left(n_{\alpha,k} + n_{\alpha,l}\right)``
and
``\bar{n}_{kl} := \bar{n}_{a,kl} + \bar{n}_{b,kl}``.
"""

# ╔═╡ a6c2f755-bdda-471a-8ca1-9801a29c8003
md"""
We use the explicit formula
```math
\begin{align}
\begin{bmatrix}
a & b \cr
c & d \cr
\end{bmatrix}^{-1} 
= 
\frac{1}{ad - bc}
\begin{bmatrix}
d & -b \cr
-c & a \cr
\end{bmatrix}
\end{align}
```
to implement the inverse ``M^{-1, \text{num}}_{kl}`` below.
"""

# ╔═╡ 12812a74-80e9-11eb-1729-d3351ff098a0
md"""
-------------------------------------------------------------------------------------
# Implementation in VoronoiFVM.jl
"""

# ╔═╡ 7f675aad-c74e-4e04-a673-8d8eea759b64
begin
	# physical constants
    const kB = 1.3806503e-23 # J/K/#
    const hbar= 1.054571817e-34 # Js
	const N_Avo = 6.02214076e23 # #/mol
	const Rgas = kB*N_Avo # J/K/mol
	# indexing of unknowns
    const iT, ia, ib = 1,2,3
	# 
	const porosity_turtuosity_ratio = 0.5 # 1
	# effective diffusion coefficients
	const Dab, DaD, DbD =  porosity_turtuosity_ratio*[1e-6, 2e-6, 3e-6] # m^2/s
	# heat conductivity of dust
	const lambda = 1.0e1 # W/K/m
	# heat capacitance / R
	const cVD = 1.0 # dust
    const cV = [0.0, 1.5, 2.5] # gasses [cV/R] = 1
	# heat capacity ratio
    const gamma = [0.0 , 5.0/3.0, 1.4] # 1
	# atomic masses
	const m = 1.054571817e-34*[0.0 , 20.18 , 2*14.00] # kg/molecule  #Ne, N2 
	#
	const T0 = 298.15 # K ~ 25 grad C
	const p0 = 1.01e5 # Pa
end;

# ╔═╡ 11294978-2b7d-41be-aaea-56f4d4054dfa
phi_factor(index) = 4*pi*exp(gamma[index])*m[index]/3.0/hbar^2

# ╔═╡ 6f531a7d-b5e6-400a-8d76-46210baaacbf
Phi = [0.0 , phi_factor(ia), phi_factor(ib)];

# ╔═╡ 53584a9c-d919-471b-8068-a792766c0f87
md"""
| 			 		| ``c^V`` 	  	| ``\gamma``   |
|:---------- 		| ---------- 	|:------------:|
| monoatomic ``a``  | $(cV[ia])``k_B`` |      $(gamma[ia])  | 
| diatomic ``b``    | $(cV[ib])``k_B`` | $(gamma[ib])	   |
"""

# ╔═╡ dc7c808e-4f9c-4618-9530-f1ad4603c5f1
md"""
## System
"""

# ╔═╡ e24acf23-a353-4578-aaa1-17ff53fafd22
md"""
Grid
"""

# ╔═╡ d871263b-6c38-49d1-8978-9ee11ae216f6
grid_p_num=50

# ╔═╡ 1e964546-d463-4234-8091-959fcc5fdb40
grid=VoronoiFVM.Grid(collect(0:1/convert(Float64,grid_p_num):1))

# ╔═╡ 05601e09-179c-4b48-abb2-0b203a21b288
md"""
Implementation of the time derivatives
"""

# ╔═╡ 6cbcae74-80d7-11eb-2dc7-cdbbc8c9e70f
function storage!(f,u,node)
f[ia]=u[ia]
f[ib]=u[ib]
f[iT]=kB*N_Avo*(
		 cV[ia]*u[ia]
   		+cV[ib]*u[ib]
   		+cVD
	)*u[iT]
end

# ╔═╡ bf9bd080-465f-464d-9d57-d91e73a91cc1
md"""
Discrete system constructor
"""

# ╔═╡ b32de11a-80d8-11eb-2d2a-7f9b90df752d
function discrete_system(fluxes)
	physics=VoronoiFVM.Physics(num_species=3,
                               flux=fluxes,
                               storage=storage!
                               )
    sys=VoronoiFVM.System(grid,physics)
	for i in [iT,ia, ib]
			enable_species!(sys,i,[1]) 
	end
	return sys
end

# ╔═╡ ea8dae77-d3d5-4c1b-bf66-8f8713deb1dd
md"""
The system is endowed with Dirichlet boundary conditions for all the unknowns
"""

# ╔═╡ 8b4b5965-036d-4281-97a1-ca98c7b01542
md"""
```math
\begin{align*}
T &\quad = \quad T
~, \\
n_a = x_a \frac{p}{RT} &\quad\leftrightarrow\quad x_a = \frac{n_a}{n_a+n_b}
~,\\
n_b = (1-x_a) \frac{p}{RT}& \quad\leftrightarrow\quad  p = (n_a+n_b) R T
~.
\end{align*}
```
"""

# ╔═╡ 03afba43-51f0-49e6-9bb3-f4fd1d6c44d6
Tnanb(X) =  [
	# X[1] - temperature
	# X[2] - x_a
	# X[3] - pressure/p0
	X[1], 								# temperature
	X[2]        *X[3]*p0/(Rgas*X[1]),   # n_a
	(1.0 - X[2])*X[3]*p0/(Rgas*X[1]),   # n_b
]

# ╔═╡ e163f524-98e9-4e7c-9a88-ea0843afca18
Txap(U) =  (
	U[iT,:], 								# temperature 
	U[ia,:]./(U[ia,:].+U[ib,:]), 			# x_a
	(U[ia,:].+U[ib,:]).*U[iT,:]./p0*Rgas 	# p
)

# ╔═╡ d6d275c1-1e54-427b-9bcd-d7139390c4be
function Dbcs!(sys, bval_left, bval_right)
	for (iboundary, bvalues) in zip([1,2], [bval_left, bval_right])
		for (item, bval) in enumerate(bvalues)
			boundary_dirichlet!(sys, item, iboundary, bval)
		end
	end
end

# ╔═╡ b4aa8fb7-f5d8-45d9-9df8-dff76500075e
set_Txap_Dbcs!(sys, ulr) = Dbcs!(sys, Tnanb(ulr[1]), Tnanb(ulr[2]))

# ╔═╡ ab6b6b29-bd62-4404-9b3e-840cab764a1b
md"""
## Implementation of force-flux relations
"""

# ╔═╡ 7d1787be-bc43-410c-a3d5-e8d34ede2508
md"""
### Implicit fluxes -- automatic differentiation of the backslash operator
The numerical fluxes $\textbf{J}^{\textrm{num}}_{kl}$ are computed using the backslash operator (Julia solver for linear systems) 

$\textbf{J}^{\textrm{num}}_{kl} 
= 
M^{\textrm{num}}_{kl} 
\ \backslash_\textrm{Julia}\ 
\textbf{F}^{\textrm{num}}_{kl}$

"""

# ╔═╡ 5dd552bb-8de2-4074-9924-25e0204edc6f
function implicitJ(Fa, Fb, nae, nbe, ne)
	M = @SArray[-(nbe/ne/Dab + 1/DaD)      nae/ne/Dab;
         nbe/ne/Dab                 -(nae/ne/Dab + 1/DbD)]
    return M\@SArray[Fa, Fb] 
end

# ╔═╡ d82853d0-827f-11eb-15d1-bbc6ed16997d
md"
### Explicit fluxes
The numerical fluxes $\textbf{J}^{\textrm{num}}_{kl}$ are computed as product of discretized inverse $M^{-1,\textrm{num}}_{kl}$ and the discretization of the forces as

$\textbf{J}^{\textrm{num}}_{kl} 
= 
M^{-1,\textrm{num}}_{kl} 
\textbf{F}^{\textrm{num}}_{kl}$
"

# ╔═╡ 70472b0e-57aa-46d4-b123-2e667402781c
function explicitJ(Fa, Fb, nae, nbe, ne)
	detM = (1/DaD/DbD + nae/ne/DaD/Dab + nbe/ne/DbD/Dab)
    Minv = detM^(-1)*@SArray[-(nae/ne/Dab + 1/DbD)      -nae/ne/Dab;
         			  -nbe/ne/Dab                -(nbe/ne/Dab + 1/DaD)]
	return Minv*@SArray[Fa,Fb]
end

# ╔═╡ 9fddf8d0-3623-43b8-b4d6-af77beff918e
md"""
###### Auxiliary
"""

# ╔═╡ 9e051e22-c000-4f55-9d71-d4e737b6015d
density_means(u) = 0.5*(u[ia,1] + u[ia,2]), 0.5*(u[ib,1] + u[ib,2])

# ╔═╡ e648e5be-5ed8-4329-8061-0c81cd86634c
function sedan(g, u1, u2)
	# flux: j = - ( u' + u*q)
	# dirichlet BCs u1        u2 
	# line segment  |---------|
	# colloc. pts  x1        x2
	# h = |x2-x1|
	# g = q*h
  	bp,bm=fbernoulli_pm(g) # Bernoulli function for B(+g) and B(-g)
  	return (-1.0)*( u2*bm - u1*bp )
end

# ╔═╡ ba43a804-7432-4035-a042-114e452c10cd
function fluxes!(f,u,edge,J::TJ) where TJ<:Function
	#
	g_ab = log(u[iT,2])-log(u[iT,1])
	#
	F_a = -1*cV[ia]*(gamma[ia]-1)*sedan(g_ab, u[ia,1], u[ia,2])
    F_b = -1*cV[ib]*(gamma[ib]-1)*sedan(g_ab, u[ib,1], u[ib,2])
	#
	n_ae, n_be = density_means(u)
    n_e  = n_ae + n_be 
	# computation of fluxes J 
    Ja, Jb = J(F_a, F_b, n_ae, n_be, n_e) 
	# 
    f[ia] = Ja 
    f[ib] = Jb
	
    g_T = kB*N_Avo/lambda*(gamma[ia]*cV[ia]*f[ia] + gamma[ib]*cV[ib]*f[ib])
    f[iT]= lambda*sedan(-g_T, u[iT,1], u[iT,2])
end

# ╔═╡ 7d30658e-20a9-42ca-be11-c60a7f9cb559
imfluxes!(f,u0,edge) = fluxes!(f,u0,edge,implicitJ)

# ╔═╡ de52a931-9dbf-4fb2-ac09-e1f4fab8a581
exfluxes!(f,u0,edge) = fluxes!(f,u0,edge,explicitJ)

# ╔═╡ 0d521e4a-732e-4590-ba76-233b0944c287
md"""
# Results
"""

# ╔═╡ daf3a39d-a987-4609-951a-b6bfeea1132f
md"""
## Stationary solution
"""

# ╔═╡ 324ed7aa-235c-455d-a2b0-472e7c793b86
function constant_initial_values_setter!(inival, sys)
	Dbcs = sys.boundary_values
	coord=coordinates(sys.grid)
    @views inival[ia,:] .= Dbcs[ia,1]
	@views inival[iT,:] .= Dbcs[iT,1]
    @views inival[ib,:] .= Dbcs[ib,1]
end

# ╔═╡ 34b29da8-01f6-44ac-bd46-add6318e0f1d
function initial_values_setter!(inival, sys)
	Dbcs = sys.boundary_values
	coord=coordinates(sys.grid)
    for (i,x) in enumerate(coord)
        @views inival[ia,i] = Dbcs[ia,1]   + (Dbcs[ia,2] - Dbcs[ia,1])*x
		@views inival[iT,i] = Dbcs[iT,1]   + (Dbcs[iT,2] - Dbcs[iT,1])*x
        @views inival[ib,i] = Dbcs[ib,1]   + (Dbcs[ib,2] - Dbcs[ib,1])*x
    end
end

# ╔═╡ 283d57f2-80dd-11eb-3458-b37ebff574de
function stationary_solution(fluxes, Dbcs_rl)
	sys = discrete_system(fluxes)
	set_Txap_Dbcs!(sys, Dbcs_rl)
	#
	inival = VoronoiFVM.unknowns(sys)
    constant_initial_values_setter!(inival, sys)
    U= VoronoiFVM.unknowns(sys)
	#
    VoronoiFVM.solve!(U,inival,sys)
	return sys, U
end

# ╔═╡ 4745e3aa-6df3-47dd-b7e9-0cde0961aa95
md"""
## Steady state: implicit vs. explicit fluxes
"""

# ╔═╡ 73cac13b-1a39-4010-ae45-3f6c375fa7a6
function plot_ss2(sys, U1, U2 ;t1="implicit", t2="explicit")
	#coord=coordinates(sys.grid)
	X = sys.grid.components[XCoordinates]
	#PyPlot.rcParams["text.usetex"] = True

	mt = @sprintf("%s (solid) vs. %s (dashed)",t1,t2)
	figss2, axes = PyPlot.subplots(2,1)
	#
	T1,xa1,p1 = Txap(U1)
	T2,xa2,p2 = Txap(U2)
	#
	PyPlot.clf()
    
	PyPlot.subplot(211)
	
	PyPlot.plot(X,T1, color=:blue, label="T")
	PyPlot.plot(X,T2, color=:blue, ls="dotted", label="T")
    PyPlot.ylabel("\$T/K\$")
	PyPlot.grid()
	PyPlot.title("Comparison: "*t1*" (solid) vs "*t2*" (dotted)")
    PyPlot.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
	#
	PyPlot.subplot(212)
	PyPlot.plot(X,xa1, color=:red, label =L"x_a")
	PyPlot.plot(X,xa2, color=:red, ls="dotted", label=L"x_a")
	PyPlot.plot(X,p1, color=:green, label=L"p/p^{ref}")
	PyPlot.plot(X,p2, color=:green, label=L"p/p^{ref}", ls="dotted")
	PyPlot.xlabel("\$x/m\$")
	PyPlot.ylabel("\$[1]\$")
	PyPlot.grid()
	PyPlot.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
	figss2.tight_layout()
	
	figss2
end

# ╔═╡ 357bce77-eb5a-45c9-b2dc-f3bae6ea5598
md"""
## Map of losses
"""

# ╔═╡ 5e6a5dc8-66b1-4335-829a-4bc348886e45
md"""
The stationary energy balance can be split into the flux of the Gibbs energy ``\dot\Delta G`` and the losses. The density of the losses, denoted ``\textrm{MOL}(x)``, is the correct optimization functional in non-isothermal scenarios.
"""

# ╔═╡ 8a83dae5-9e86-4a71-971e-04422f96761d
md"""
```math
\begin{align}
0 =&
\int_\Omega
\partial_i\left(
	-\lambda\partial_i T
	+T\frac{s_a}{n_a}J^i_a
	+T\frac{s_b}{n_b}J^i_b
\right)
+
\partial_i\left(
	\mu_a J^i_a
	+\mu_b J^i_b
\right)
\\
=& 
\int_\Omega
\partial_i\left(
	T\left[
		-\lambda\frac{\partial_i T}{T}
		+\frac{s_a}{n_a}J^i_a
		+\frac{s_b}{n_b}J^i_b
	\right]
\right)
+
\partial_i\left(
	\mu_a J^i_a
	+\mu_b J^i_b
\right)
\\
=& 
\underbrace{
	\int_{\partial\Omega}
	\left(
		\mu_a J^i_a
		+\mu_b J^i_b
	\right)
	\cdot \nu_i
}_{\dot\Delta G}
+
\int_\Omega
\underbrace{
	\left[
	\partial_i
		T\overbrace{\left(
			-\lambda\frac{\partial_i T}{T}
			+\frac{s_a}{n_a}J^i_a
			+\frac{s_b}{n_b}J^i_b
		\right)	
		}^{j_{s, tot}}
		+
		T\sigma
	\right]
}_{=:\,\textrm{MOL}(x)}
\end{align}
```
"""

# ╔═╡ dd832942-ac9a-40ad-9ab1-025cebf22269
md"""
The map of losses can be defined as 
``MOL(x) = \nabla T\cdot j_\textrm{s, tot} + T\sigma``
"""

# ╔═╡ 4568fe56-80e4-11eb-3202-81d66c380665
begin
	p1_slider = @bind p_val_1 html"<input type='range' min='.50' max='2.0' step='0.05' value='1.0'>"
	p2_slider = @bind p_val_2 html"<input type='range' min='.50' max='2.0' step='0.05' value='1.0'>"
	xa1_slider = @bind xa_val_1 html"<input type='range' min='0.0' max='1.0' step='0.05' value='0.5'>"
	xa2_slider = @bind xa_val_2 html"<input type='range' min='0.0' max='1.0' step='0.05' value='0.5'>"
	T1_slider = @bind T_b_1 html"<input type='range' min='-10.0' max='10.0' step='0.5' value='0.0'>"
	T2_slider = @bind T_b_2 html"<input type='range' min='-10.0' max='10.0' step='0.5' value='0.0'>"
	
	md"""**Dirichlet boundary values:**
	
	total pressure ``p=(n_a+n_b)k_B T``: left  $(p1_slider) right $(p2_slider)
	
	molar fraction ``x_a=\frac{n_a}{n_a+n_b}`` left: $(xa1_slider) right: $(xa2_slider)
	
	``\Delta T`` left: $(T1_slider) right: $(T2_slider)
	"""
end

# ╔═╡ 43cdb9e6-a20e-418b-8745-f922ab06e549
function Dirichlet_BCs!(sys)
	bval_left = Tnanb([T0 + T_b_1, xa_val_1, p_val_1])
	bval_right= Tnanb([T0 + T_b_2, xa_val_2, p_val_2])
	Dbcs!(sys, bval_left, bval_right)
end

# ╔═╡ e8a5f8d5-0c03-4ddb-b2d1-3b9e56dccbd5
stationaryDbcs = ([T0 + T_b_1, xa_val_1, p_val_1], [T0 + T_b_2, xa_val_2, p_val_2])

# ╔═╡ fbed530c-e82d-47f7-8603-34b3092e8a56
sys_im, U_im = stationary_solution(imfluxes!, stationaryDbcs)

# ╔═╡ cf506e4a-95ae-4e3d-a05d-ee825fb111b8
sys_ex, U_ex = stationary_solution(exfluxes!, stationaryDbcs)

# ╔═╡ c75d60dc-ac14-4dab-ac50-ae9d03ff1758
plot_ss2(sys_im, U_im, U_ex)

# ╔═╡ eb0aa7c2-0db1-4028-9bb3-e9d0ed0fc26c
md"""
| Boundary conditions								| left	  		| right   	   |
|:---------- 			 							| ---------- 	|:------------:|
| pressure ``p/p^\textrm{ref}``  					| $(p_val_1) 	| $(p_val_2)   | 
| molar fraction ``x_a``    						| $(xa_val_1) 	| $(xa_val_2)  |
| temperature ``273.15\,\textrm{K}+\Delta T``    	| $(T_b_1) 		| $(T_b_2)	   |
"""

# ╔═╡ f0d2c7e0-aa9e-40e9-bffe-c3d2adf71f2e
md"""
## Temperature step relaxation
"""

# ╔═╡ 820c01b0-70d3-4ff2-a936-a373ec51b16f
begin
	pTl_sl = @bind pTl html"<input type='range' min='.50' max='2.0' step='0.05' value='1.0'>"
	pTr_sl = @bind pTr html"<input type='range' min='.50' max='2.0' step='0.05' value='1.0'>"
	xaTl_sl = @bind xaTl html"<input type='range' min='0.0' max='1.0' step='0.05' value='0.5'>"
	xaTr_sl = @bind xaTr html"<input type='range' min='0.0' max='1.0' step='0.05' value='0.5'>"
	TTl_sl = @bind TTl html"<input type='range' min='-10.0' max='10.0' step='0.5' value='0.0'>"
	TTr_sl = @bind TTr html"<input type='range' min='-10.0' max='10.0' step='0.5' value='0.0'>"
	
	md"""**Dirichlet boundary values for the temperature step problem:**
	
	total pressure ``p=(n_a+n_b)k_B T``: left  $(pTl_sl) right $(pTr_sl)
	
	molar fraction ``x_a=\frac{n_a}{n_a+n_b}`` left: $(xaTl_sl) right: $(xaTr_sl)
	
	``\Delta T`` left: $(TTl_sl) right: $(TTr_sl)
	"""
end

# ╔═╡ 52d7e225-c64e-4e91-98f1-ac9753c0ac82
Tstep_Dbcs = ([T0+TTl, xaTl, pTl],[T0+TTr, xaTr, pTr])

# ╔═╡ 5ee11544-b6ca-44d2-9827-822bbb26147a
function temperature_step_evolution(T_step)
	tend = 5.0e5
	#
	control=VoronoiFVM.NewtonControl()
    #control.damp_initial = 1e-5
    #control.damp_growth = 1.4
	#control.verbose = true
	control.handle_exceptions = true
	control.Δt=1.0e-4   
	control.Δt_min=1.0e-50    
	control.Δt_max=0.1*tend # allow to increase the time step
	control.Δu_opt=0.05 # this actually is used to control the timestep sitze
	control.tol_round=1.0e-11 # what change of |u|_1 is within roundoff error
	control.max_round=3 # believe the above only after 3 repeated cases
	control.tol_relative = 1e-9
	#
	sys, steady = stationary_solution(imfluxes!, Tstep_Dbcs)
	#check_allocs!(sys,false)
	#
	T0 = sys.boundary_values[iT,1]
	T1, nal, nbl = Tnanb([T0 + T_step, Tstep_Dbcs[1][2], Tstep_Dbcs[1][3]])
	boundary_dirichlet!(sys,iT,1, T1) # temperature step
	boundary_dirichlet!(sys,ia,1, nal) # temperature step
	boundary_dirichlet!(sys,ib,1, nbl) # temperature step
	#
	U=VoronoiFVM.solve(steady,sys, [0.0,tend], control=control)
	U_stdy = VoronoiFVM.solve(steady,sys, control=control)
	return sys, U, U_stdy
end

# ╔═╡ 0463bbc2-55f4-48c8-a11d-47dd04a58adc
sys_evol, sol_evol, sol_stdy = temperature_step_evolution(200.0);

# ╔═╡ 16daab72-3736-48db-a98d-004b572c6272
md"""
### Temperature step end time
"""

# ╔═╡ 32266f89-abde-42cb-b14d-5c339948c3ba
plot_ss2(sys_evol,sol_evol.u[end], sol_stdy,t1="evolution-end",t2="steady")

# ╔═╡ 8cc29824-126b-4262-b3b3-32e0566ea439
relative_error = maximum((sol_evol.u[end] .- sol_stdy)./sol_stdy)

# ╔═╡ e40bcd4a-ca0f-4a77-b1da-42d84cf1e6e0
 mysavefig(figname)=PyPlot.savefig(figname);

# ╔═╡ c3bf6d67-7f82-43ae-aac7-c8ea5e9a0296
gridfunc(X, T, U) = (X, log10.(T .+ 1.0e-20), transpose(reshape(U, length(X), length(T))));

# ╔═╡ 722612e3-a57b-4242-ba7d-316ad00613c5
function plot_evolution(sys, sol)
	cutoff = 5*10^3
	X = sys.grid.components[XCoordinates]
	t = sol.t[cutoff:end]
	Temp = ElasticArray{Float64}(undef, length(X), 0)
	xa = ElasticArray{Float64}(undef, length(X), 0)
	p = ElasticArray{Float64}(undef, length(X), 0)
	for i=cutoff:length(sol)
		T, xxa, pp = Txap(sol.u[i])
		append!(Temp, T)#sol.u[i][1,:])
		append!(xa, xxa)#sol.u[i][2,:]./(sol.u[i][2,:]+sol.u[i][3,:]))
		append!(p, pp)#(sol.u[i][2,:]+sol.u[i][3,:])*kB*N_Avo.*sol.u[i][1,:]/p0)
	end
	
	Tmax, Tmin = maximum(Temp), minimum(Temp)
    fig2, axes = PyPlot.subplots(num=2, figsize=[10,4], nrows=1, ncols=3)
    PyPlot.clf()
    #
	PyPlot.subplot(131)
    PyPlot.title(@sprintf("\$T\\in\$[%.2f,%.2f]",Tmax,Tmin))
    vTemp = collect(Tmin : (Tmax-Tmin)/100 : Tmax)
    tTemp = collect(Tmin: (Tmax-Tmin) / 10:Tmax)
    cntTemp = PyPlot.contourf(gridfunc(X, t, Temp)..., vTemp, cmap=ColorMap("hot"))
    for c in cntTemp.collections
        c.set_edgecolor("face")
    end
    PyPlot.contour(gridfunc(X, t, Temp)..., colors="k", tTemp)
    PyPlot.xlabel("\$x\$")
    PyPlot.ylabel("\$\\log_{10}(t)\$")

    Cmax = 1.0
    xav = collect(0:Cmax / 100:Cmax)
    xat = collect(0:Cmax / 8:Cmax)
    #
    PyPlot.subplot(132)
    PyPlot.gca().get_yaxis().set_visible(false)
	PyPlot.title(@sprintf("\$x_a\\in\$(%.2f,%.2f)",minimum(xa),maximum(xa)))
	cntxa = PyPlot.contourf(gridfunc(X, t, xa)..., xav, cmap=ColorMap("terrain"))
    for c in cntxa.collections
        c.set_edgecolor("face")
    end
    PyPlot.contour(gridfunc(X, t, xa)..., colors="k", xat)
    PyPlot.xlabel("\$x\$")
	
	pmin, pmax = minimum(p),maximum(p)
    pv = collect(pmin: (pmax-pmin) / 100:pmax)
    pt = collect(pmin: (pmax-pmin) / 8:pmax)
	#
    PyPlot.subplot(133)
    PyPlot.gca().get_yaxis().set_visible(false)
	PyPlot.title(@sprintf("\$p/p_{ref}\\in\$(%.2f,%.2f)",minimum(p),maximum(p)))
	cntp = PyPlot.contourf(gridfunc(X, t, p)..., pv, cmap=ColorMap("terrain"))
    for c in cntp.collections
        c.set_edgecolor("face")
    end
    PyPlot.contour(gridfunc(X, t, p)..., colors="k", pt)
    PyPlot.xlabel("\$x\$")

	
    cax = fig2.add_axes([0.01, 0.0, 0.1, 1.0])
    cax.axis("off")
    fig2.colorbar(cntTemp, ax=cax, ticks=tTemp, boundaries=vTemp, location="left", label="T[K]") 
	
	cax = fig2.add_axes([0.9, 0.0, 0.1, 1.0])
    cax.axis("off")
	fig2.colorbar(cntp, ax=cax, ticks=pt, boundaries=pv, label="p[bar]") 
	
	cax = fig2.add_axes([0.74, 0.0, 0.1, 1.0])
    cax.axis("off")
	fig2.colorbar(cntxa, ax=cax, ticks=xat, boundaries=xav, label="\$x_{a}\$") 
	
	mysavefig("f2.svg")
	fig2
end

# ╔═╡ 576bdecb-bae4-4976-87ca-e149124b153f
plot_evolution(sys_evol, sol_evol)

# ╔═╡ b57c671d-c2cf-43c5-8ab0-386b0c6ebd15
md"""
#### Supporting functions for MOL calculation
"""

# ╔═╡ 0aa543cf-cc4e-4047-a9a4-772c1c55b2ee
md"""
##### Thermodynamic potentials
"""

# ╔═╡ 1986eec0-7498-4f25-a2bd-9c769ce2605b
md"""
```math
s_\alpha(n_\alpha, T)
  = n_\alpha c_\alpha^V \ln \left({c_\alpha^V n_\alpha^{1-\gamma_\alpha} T\Phi_\alpha}\right)
  ~,
```
"""

# ╔═╡ 4ecb6c78-c34d-4b2c-80cc-c470c555894f
md"""
```math
\mu_\alpha(n_\alpha, T)
  = c_\alpha^V T \left(  
      \gamma_\alpha - \ln \left({c_\alpha^V n_\alpha^{1-\gamma_\alpha}T\Phi_\alpha}\right)
    \right)
  ~,
```
"""

# ╔═╡ 7552b0d0-2c68-415a-b942-1824bafde87b
function entropies_per_mol(U) 
	partial_entropies = similar(U)
	for index in [ia, ib]
		# s_a/n_a
		partial_entropies[index,:] = Rgas*cV[index]*U[index,:].*log.(
			Rgas*cV[index]*U[index,:].^(1.0-gamma[index]).*U[iT,:].*Phi[index]
		)
	end
	partial_entropies
end;

# ╔═╡ 7f4bbd0a-43f7-442e-8c4e-dd0ee75f0f53
function chemical_potentials_per_mol(U)
	entropies = entropies_per_mol(U)
	mu = similar(U)
	for index in [ia, ib]
		mu[index,:] = U[iT,:].*(
			Rgas*cV[index]*gamma[index]
			.-
			entropies[index,:]./U[index]
		)
	end
	mu
end;

# ╔═╡ 92706c87-559c-4ad2-b950-50a384a5d170
function potentials_plot(sys, U)
	entropies = entropies_per_mol(U)
	mu = chemical_potentials_per_mol(U)
	coord = coordinates(sys.grid)
	p=Plots.plot(grid=true, title="Chemical potentials, partial molar entropies", ylabel=L"\textrm{J/mol}", legend=:right)
    @views begin
		Plots.plot!(p,coord[1,:],U[iT,:].*entropies[ia,:]./U[ia,:], label=L"T\frac{s_a}{n_a} ", color=:red,lw=3)
		#
		Plots.plot!(p,coord[1,:],mu[ia,:], label=L"\mu_a", color=:red,lw=3,linestyle=:dot)
		#
		Plots.plot!(p,coord[1,:],U[iT,:].*entropies[ib,:]./U[ib,:], label=L"T \frac{s_b}{n_b} ", color=:green,lw=3)
		#
		Plots.plot!(p,coord[1,:],mu[ib,:], label=L"\mu_b ", color=:green,lw=3,linestyle=:dot)
		#
    end
end;

# ╔═╡ dcaf19c7-0caa-48fa-85ce-cf762525fea5
potentials_plot(sys_im, U_im)

# ╔═╡ 0ab518a5-be36-4862-9071-275e0653b1a6
md"""
Total entropy flux
```math
\begin{align}
j_q = -\lambda\frac{\partial_i T}{T}
~,
\\
j_{s,a} = \frac{s_a}{n_a}J^i_a
~,
\\
j_{s,b} = \frac{s_b}{n_b}J^i_b
~,
\\
j_{s,tot} = j_q + j_{s,a} + j_{s,b}
~.
\end{align}
```
"""

# ╔═╡ d1e59e39-c1fb-4e71-b035-e73300bebdd8
md"""
Entropy production density
```math
\begin{align}
\sigma_\text{Fourier} 
=&\ 
\lambda\left(
	\frac{\partial_i T}{T}
\right)^2
\\
\sigma_\text{MS}
=&\
\frac{n_a n_b}{n_a+n_b}
\frac{R}{D_{ab}}
\left(
	\frac{J_a^i}{n_a}
	-
	\frac{J_b^i}{n_b}
\right)^2
\\
\sigma_\text{dust}
=&\
{n_a}
\frac{R}{D_{aD}}
\left(
	\frac{J_a^i}{n_a}
\right)^2
+
{n_b}
\frac{R}{D_{bD}}
\left(
	\frac{J_b^i}{n_b}
\right)^2
\\
\sigma 
=&\ \sigma_\text{Fourier} + \sigma_\text{MS} + \sigma_\text{dust}
\end{align}
```
"""

# ╔═╡ 9a13836a-fad0-4b96-86af-c1aff2ea5c0d
function boundary_fluxes(sys, U; side=:L)
	tf = VoronoiFVM.TestFunctionFactory(sys)
	T = testfunction(tf, 1, 2)
	I=integrate(sys,T,U)
	return I
end;

# ╔═╡ 5e74b6b6-a119-43d1-b770-ea19fc627553
function solution_derivative(sys,U)
	der = copy(U)
	coord = coordinates(sys.grid)
	gridlen = ExtendableGrids.num_nodes(sys.grid)
	for i in 1:gridlen
		if i < gridlen
			df = U[iT,i+1] - U[iT,i]
			dx = coord[i+1] - coord[i]
			der[iT,i] = df/dx
		else
			der[iT,i] = der[iT,i-1]
		end
	end
	return der
end;

# ╔═╡ d129b33f-2cff-4cd7-9b77-28d8f83f0685
function entropy_flux_contributions(sys, U)
	dU = solution_derivative(sys, U)
	I = boundary_fluxes(sys, U)
	entropies = entropies_per_mol(U)
	return -lambda*(dU[iT,:]./U[iT,:]), I[ia]*entropies[ia,:]./U[ia,:], I[ib]*entropies[ib,:]./U[ib,:]
end;

# ╔═╡ 3acafa9b-7c12-4e6e-98a6-f3083d05f571
function entropy_flux_contr_plot(sys, U)
	jsT, jsa, jsb = entropy_flux_contributions(sys, U)
	coord = coordinates(sys.grid)
	p=Plots.plot(grid=true, title="Entropy flux contributions", ylabel=L"\textrm{J/K/m^2/s}")
    @views begin
		Plots.plot!(p,coord[1,:],jsT, label=L"j_q", color=:cyan)
		Plots.plot!(p,coord[1,:],jsa, label=L"j_{s,a}", color=:magenta)
		Plots.plot!(p,coord[1,:],jsb, label=L"j_{s,b}", color=:black)
	end
end;

# ╔═╡ aee68c17-fa0a-441f-b8ac-aaea43564f15
entropy_flux_contr_plot(sys_im, U_im)

# ╔═╡ e56376c3-eafb-4386-8305-b37b97f73859
function dissipation_density(sys, U)
	dU = solution_derivative(sys, U)
	I = boundary_fluxes(sys, U)
	sigma_Fourier = lambda*(dU[iT,:]./U[iT,:]).^2
	sigma_MS = Rgas*U[ia,:].*U[ib,:]./(U[ia,:].+U[ib,:])/Dab.*(
		I[ia]./U[ia,:]
		-
		I[ib]./U[ib,:]
	).^2
	sigma_dust = Rgas*(
		U[ia,:]./DaD.*(I[ia]./U[ia,:]).^2
	   	.+
		U[ib,:]./DbD.*(I[ib]./U[ib,:]).^2
	)
	return sigma_Fourier, sigma_MS, sigma_dust
end;

# ╔═╡ 7c64feb1-ab36-4211-a2a2-9bc3110b2d6c
function map_of_losses(sys, U)
	dU = solution_derivative(sys, U)
	I = boundary_fluxes(sys, U)
	jsT, jsa, jsb = entropy_flux_contributions(sys, U)
	sigma_Fourier, sigma_MS, sigma_dust = dissipation_density(sys, U)
	Tsigma = U[iT,:] .* (sigma_Fourier + sigma_MS + sigma_dust)
	dTjs = dU[iT,:] .* (jsT .+ jsa .+jsb)
	return Tsigma, dTjs
end

# ╔═╡ de651a1b-79a9-4a41-af60-8a8275e1bd5a
function mol_plot(sys, U)
	Tsigma, Tjs = map_of_losses(sys, U)
	coord = coordinates(sys.grid)
	p=Plots.plot(grid=true, title="Map of losses", ylabel=L"\textrm{W/m^3}", legend=:top, xlabel=L"\textrm{x/m}")
    coord=coordinates(sys.grid)
    @views begin
		Plots.plot!(p,coord[1,:],Tsigma, label=L"T\sigma", color=:blue)
		Plots.plot!(p,coord[1,:],Tjs, label=L"j_\textrm{s, tot}\cdot\nabla T", color=:red)
		Plots.plot!(p,coord[1,:], Tsigma+Tjs, label=L"\textrm{MOL}", color=:green)
    end
end

# ╔═╡ 1bde0cbd-b551-49d5-846e-84d6a44f0d0b
mol_plot(sys_im, U_im)

# ╔═╡ 759bf470-7044-477e-88a2-830cd0023cbb
function dissipation_density_plot(sys, U)
	sigma_Fourier, sigma_MS, sigma_dust = dissipation_density(sys, U)
	coord = coordinates(sys.grid)
	p=Plots.plot(grid=true, title="Entropy production contributions", ylabel=L"\textrm{J/K/m^3/s}")
    @views begin
		Plots.plot!(p,coord[1,:],sigma_Fourier, label=L"\sigma_{\textrm{Fourier}}", color=:blue)
		Plots.plot!(p,coord[1,:],sigma_MS, label=L"\sigma_\textrm{MS}", color=:red)
		Plots.plot!(p,coord[1,:],sigma_dust, label=L"\sigma_\textrm{dusty}", color=:green)
    end 
end;

# ╔═╡ 157b6a2d-31d1-4a49-a93b-ff5da918a9dd
dissipation_density_plot(sys_im, U_im)

# ╔═╡ 8bb4876b-482e-4ab3-aa39-e35c1443c4b0
function temp_grad(sys, U)
	dU = solution_derivative(sys, U)
	I = mass_fluxes(sys, U)
	enthalpy_flux = I[ia]*Rgas*cV[ia]*gamma[ia] + I[ib]*Rgas*cV[ib]*gamma[ib]
	return -lambda*dU[iT,:], I[iT].*ones(size(U[iT,:]))
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ElasticArrays = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
ExtendableGrids = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GridVisualize = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
VoronoiFVM = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"

[compat]
ElasticArrays = "~1.2.9"
ExtendableGrids = "~0.9.5"
ForwardDiff = "~0.10.26"
GridVisualize = "~0.5.1"
LaTeXStrings = "~1.3.0"
Plots = "~1.27.6"
PlutoUI = "~0.7.38"
PyPlot = "~2.10.0"
StaticArrays = "~1.4.7"
VoronoiFVM = "~0.16.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc1"
manifest_format = "2.0"
project_hash = "666150fbeaf4955a7de11cf3803b81b27c0f646f"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "f4a6ecff7407a29d5d15503508144b7cc81bdc63"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.25.3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "c933ce606f6535a7c7b98e1d86d5d1014f730596"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.7"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5a4168170ede913a2cd679e53c2123cb4b889795"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.53"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "5f5f0b750ac576bcf2ab1d7782959894b304923e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.9"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "d0fa82f39c2a5cdb3ee385ad52bc05c42cb4b9f0"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.4.5"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "a0fcc1bb3c9ceaf07e1d0529c9806ce94be6adf9"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.9"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "d064b0340db45d48893e7604ec95e7a2dc9da904"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.5.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.ExtendableGrids]]
deps = ["AbstractTrees", "Dates", "DocStringExtensions", "ElasticArrays", "InteractiveUtils", "LinearAlgebra", "Printf", "Random", "SparseArrays", "StaticArrays", "Test", "WriteVTK"]
git-tree-sha1 = "cec19e62fc126df338de88585f45a763f7601bd3"
uuid = "cfc395e8-590f-11e8-1f13-43a2532b2fa8"
version = "0.9.5"

[[deps.ExtendableSparse]]
deps = ["DocStringExtensions", "LinearAlgebra", "Printf", "Requires", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "eb3393e4de326349a4b5bccd9b17ed1029a2d0ca"
uuid = "95c220a8-a1cf-11e9-0c77-dbfce5f500b3"
version = "0.6.7"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "40d1546a45abd63490569695a86a2d93c2021e54"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.26"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.GridVisualize]]
deps = ["ColorSchemes", "Colors", "DocStringExtensions", "ElasticArrays", "ExtendableGrids", "GeometryBasics", "HypertextLiteral", "LinearAlgebra", "Observables", "OrderedCollections", "PkgVersion", "Printf", "StaticArrays"]
git-tree-sha1 = "5d845bccf5d690879f4f5f01c7112e428b1fa543"
uuid = "5eed8a63-0fb0-45eb-886d-8d5a387d12b8"
version = "0.5.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "Logging", "MultivariatePolynomials", "Primes", "Random"]
git-tree-sha1 = "18e3139ab69bfc03a8609027fd0e5572a5cffe6e"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.2.3"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "fbd884a02f8bf98fd90c53c1c9d2b21f9f30f42a"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.8.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.81.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0886d229caaa09e9f56bcf1991470bd49758a69f"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.3"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "393fc4d82a73c6fe0e2963dd7c882b09257be537"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.4.6"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3b429f37de37f1fc603cc1de4a799dc7fbe4c0b6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6f2dd1cf7a4bbf4f305a0d8750e351cb46dfbe80"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "747f4261ebe38a2bc6abf0850ea8c6d9027ccd07"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "14c1b795b9d764e1784713941e787e1384268103"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.10.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "bfe14f127f3e7def02a6c2b1940b39d0dabaa3ef"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.26.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "f03796a588eba66f6bcc63cfdeda89b4a339ce4e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.30.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "314a07e191ea4a5ea5a2f9d6b39f03833bde5e08"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.21.0"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "2114b1d8517764a8c4625a2e97f40640c7a301a7"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2bbd9f2e40afd197a1379aef05e0d85dba649951"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "bfa211c9543f8c062143f2a48e5bcbb226fd790b"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.19.7"

[[deps.Symbolics]]
deps = ["ArrayInterface", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "Groebner", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "cda7b738cd940f124d74bbcb13503cd931440f70"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.4.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "d223de97c948636a4f34d1f84d92fd7602dc555b"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.10"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "11db03dd5bbc0d2b57a570d228a0f34538c586b1"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.17"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.VoronoiFVM]]
deps = ["DiffResults", "DocStringExtensions", "ExtendableGrids", "ExtendableSparse", "ForwardDiff", "GridVisualize", "IterativeSolvers", "JLD2", "LinearAlgebra", "Parameters", "Printf", "RecursiveArrayTools", "Requires", "SparseArrays", "SparseDiffTools", "StaticArrays", "Statistics", "SuiteSparse", "Symbolics", "Test"]
git-tree-sha1 = "254b5472a9f3ec970a08b24b76cba4bf1d79f3e6"
uuid = "82b139dc-5afc-11e9-35da-9b9bdfd336f3"
version = "0.16.3"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams"]
git-tree-sha1 = "bff2f6b5ff1e60d89ae2deba51500ce80014f8f6"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.14.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.41.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─e7d187f2-ee2b-4e7c-bf33-3d4351beac3f
# ╟─a4c5960a-ece3-4f86-a761-c7f8169c7a7f
# ╠═875e4f2c-80d6-11eb-00e8-5362515a5d82
# ╟─694fcd71-4691-45bb-a213-e2d83a19adf2
# ╟─adaa6907-1895-4ed0-af29-fbeb2302f8c0
# ╟─fc63abe0-8285-11eb-11d3-43457bd58690
# ╟─2a8beb1f-c83b-464a-8619-c3a8fd05b9bf
# ╟─78e067ea-0897-40e4-9c23-85272b7bc1d8
# ╟─59e1d95e-9508-442c-b554-1dc448c41549
# ╟─ab654b48-80e8-11eb-20af-dd0d3facedc8
# ╟─dc840c0c-ac34-44da-9eac-fb2897665b6f
# ╟─0d68b27a-1d5f-4a01-ad9f-0e894b07c508
# ╟─b667d0b2-57e1-4de3-9197-776720d801cc
# ╟─5389b0b8-dfed-4f3d-bdff-943356fd147a
# ╟─3a9587bd-0beb-49eb-a074-edd7e4f89924
# ╟─5c2a32be-8179-11eb-3aa1-e569dc27c378
# ╟─ce4c6012-8280-11eb-3260-714f867c2c67
# ╟─f91b77a7-66ec-40e7-aaef-79aad206a45f
# ╟─a6c2f755-bdda-471a-8ca1-9801a29c8003
# ╟─12812a74-80e9-11eb-1729-d3351ff098a0
# ╠═7f675aad-c74e-4e04-a673-8d8eea759b64
# ╠═6f531a7d-b5e6-400a-8d76-46210baaacbf
# ╠═11294978-2b7d-41be-aaea-56f4d4054dfa
# ╟─53584a9c-d919-471b-8068-a792766c0f87
# ╟─dc7c808e-4f9c-4618-9530-f1ad4603c5f1
# ╟─e24acf23-a353-4578-aaa1-17ff53fafd22
# ╠═d871263b-6c38-49d1-8978-9ee11ae216f6
# ╠═1e964546-d463-4234-8091-959fcc5fdb40
# ╟─05601e09-179c-4b48-abb2-0b203a21b288
# ╠═6cbcae74-80d7-11eb-2dc7-cdbbc8c9e70f
# ╟─bf9bd080-465f-464d-9d57-d91e73a91cc1
# ╠═b32de11a-80d8-11eb-2d2a-7f9b90df752d
# ╟─ea8dae77-d3d5-4c1b-bf66-8f8713deb1dd
# ╠═43cdb9e6-a20e-418b-8745-f922ab06e549
# ╟─8b4b5965-036d-4281-97a1-ca98c7b01542
# ╠═03afba43-51f0-49e6-9bb3-f4fd1d6c44d6
# ╠═e163f524-98e9-4e7c-9a88-ea0843afca18
# ╠═b4aa8fb7-f5d8-45d9-9df8-dff76500075e
# ╠═d6d275c1-1e54-427b-9bcd-d7139390c4be
# ╟─ab6b6b29-bd62-4404-9b3e-840cab764a1b
# ╠═ba43a804-7432-4035-a042-114e452c10cd
# ╟─7d1787be-bc43-410c-a3d5-e8d34ede2508
# ╠═5dd552bb-8de2-4074-9924-25e0204edc6f
# ╠═7d30658e-20a9-42ca-be11-c60a7f9cb559
# ╟─d82853d0-827f-11eb-15d1-bbc6ed16997d
# ╠═70472b0e-57aa-46d4-b123-2e667402781c
# ╠═de52a931-9dbf-4fb2-ac09-e1f4fab8a581
# ╟─9fddf8d0-3623-43b8-b4d6-af77beff918e
# ╠═9e051e22-c000-4f55-9d71-d4e737b6015d
# ╠═e648e5be-5ed8-4329-8061-0c81cd86634c
# ╟─0d521e4a-732e-4590-ba76-233b0944c287
# ╟─daf3a39d-a987-4609-951a-b6bfeea1132f
# ╠═324ed7aa-235c-455d-a2b0-472e7c793b86
# ╠═34b29da8-01f6-44ac-bd46-add6318e0f1d
# ╠═283d57f2-80dd-11eb-3458-b37ebff574de
# ╠═fbed530c-e82d-47f7-8603-34b3092e8a56
# ╠═cf506e4a-95ae-4e3d-a05d-ee825fb111b8
# ╟─4745e3aa-6df3-47dd-b7e9-0cde0961aa95
# ╠═e8a5f8d5-0c03-4ddb-b2d1-3b9e56dccbd5
# ╟─73cac13b-1a39-4010-ae45-3f6c375fa7a6
# ╠═c75d60dc-ac14-4dab-ac50-ae9d03ff1758
# ╟─357bce77-eb5a-45c9-b2dc-f3bae6ea5598
# ╟─5e6a5dc8-66b1-4335-829a-4bc348886e45
# ╟─8a83dae5-9e86-4a71-971e-04422f96761d
# ╟─dd832942-ac9a-40ad-9ab1-025cebf22269
# ╠═7c64feb1-ab36-4211-a2a2-9bc3110b2d6c
# ╟─4568fe56-80e4-11eb-3202-81d66c380665
# ╟─eb0aa7c2-0db1-4028-9bb3-e9d0ed0fc26c
# ╠═1bde0cbd-b551-49d5-846e-84d6a44f0d0b
# ╠═de651a1b-79a9-4a41-af60-8a8275e1bd5a
# ╟─f0d2c7e0-aa9e-40e9-bffe-c3d2adf71f2e
# ╠═5ee11544-b6ca-44d2-9827-822bbb26147a
# ╠═0463bbc2-55f4-48c8-a11d-47dd04a58adc
# ╟─820c01b0-70d3-4ff2-a936-a373ec51b16f
# ╠═52d7e225-c64e-4e91-98f1-ac9753c0ac82
# ╟─16daab72-3736-48db-a98d-004b572c6272
# ╠═32266f89-abde-42cb-b14d-5c339948c3ba
# ╠═8cc29824-126b-4262-b3b3-32e0566ea439
# ╠═e40bcd4a-ca0f-4a77-b1da-42d84cf1e6e0
# ╠═576bdecb-bae4-4976-87ca-e149124b153f
# ╠═c3bf6d67-7f82-43ae-aac7-c8ea5e9a0296
# ╠═722612e3-a57b-4242-ba7d-316ad00613c5
# ╠═b57c671d-c2cf-43c5-8ab0-386b0c6ebd15
# ╟─0aa543cf-cc4e-4047-a9a4-772c1c55b2ee
# ╟─1986eec0-7498-4f25-a2bd-9c769ce2605b
# ╟─4ecb6c78-c34d-4b2c-80cc-c470c555894f
# ╠═7552b0d0-2c68-415a-b942-1824bafde87b
# ╠═7f4bbd0a-43f7-442e-8c4e-dd0ee75f0f53
# ╠═dcaf19c7-0caa-48fa-85ce-cf762525fea5
# ╠═92706c87-559c-4ad2-b950-50a384a5d170
# ╟─0ab518a5-be36-4862-9071-275e0653b1a6
# ╠═d129b33f-2cff-4cd7-9b77-28d8f83f0685
# ╠═aee68c17-fa0a-441f-b8ac-aaea43564f15
# ╠═3acafa9b-7c12-4e6e-98a6-f3083d05f571
# ╟─d1e59e39-c1fb-4e71-b035-e73300bebdd8
# ╠═e56376c3-eafb-4386-8305-b37b97f73859
# ╠═157b6a2d-31d1-4a49-a93b-ff5da918a9dd
# ╠═759bf470-7044-477e-88a2-830cd0023cbb
# ╠═9a13836a-fad0-4b96-86af-c1aff2ea5c0d
# ╠═5e74b6b6-a119-43d1-b770-ea19fc627553
# ╠═8bb4876b-482e-4ab3-aa39-e35c1443c4b0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
