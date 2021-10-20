struct ExplicitRungeKuttaScheme
    step_size::Float64
    alpha::LinearAlgebra.LowerTriangular{Float64}
    beta::LinearAlgebra.LowerTriangular{Float64}
    function ExplicitRungeKuttaScheme(step_size::Float64,
        alpha::LinearAlgebra.LowerTriangular{Float64},
        beta::LinearAlgebra.LowerTriangular{Float64})

        t1 = (step_size > 0)
        checksquare(alpha)
        l1 = size(alpha)
        l2 = size(beta)
        (!t1)&&throw(DomainError("step_size must be positive"))
        !(l1==l2)&&throw(DimensionMismatch("alpha, beta must have same size"))
        return new(step_size,alpha,beta)
    end
end

"""
    ForwardEuler(step_size::Float64)

Defines an Euler integration scheme to be used in `integrate_time`.

# Arguments 
-  `step_size::Float64`: the step size of the integration scheme.
"""
ForwardEuler(step_size::Float64) = ExplicitRungeKuttaScheme(
        step_size,
        LinearAlgebra.LowerTriangular([1.0][:,:]),
        LinearAlgebra.LowerTriangular([1.0][:,:])
    )

Heuns(step_size::Float64) = ExplicitRungeKuttaScheme(
        step_size,
        LinearAlgebra.LowerTriangular([1.0 0.0; 0.5 0.5]),
        LinearAlgebra.LowerTriangular([1.0 0.0; 0.0 0.5])
    )

StableRK3(step_size::Float64) = ExplicitRungeKuttaScheme(
        step_size,
        LinearAlgebra.LowerTriangular([1.0 0.0 0.0; 0.75 0.25 0.0; 1/3 0.0 2/3]),
        LinearAlgebra.LowerTriangular([1.0 0.0 0.0; 0.0 0.25 0.0; 0.0 0.0 2/3])
    )
                
_α = [  1.0                0.0                 0.0                 0.0                 0.0                 ;
        0.44437049406734   0.55562950593266    0.0                 0.0                 0.0                 ;
        0.62010185138540   0.0                 0.37989814861460    0.0                 0.0                 ;
        0.17807995410773   0.0                 0.0                 0.82192004589227    0.0                 ;
        0.00683325884039   0.0                 0.51723167208978    0.12759831133288    0.34833675773694    ]
_β = [  0.39175222700392    0.0                 0.0                 0.0                 0.0                 ;
        0.0                 0.36841059262959    0.0                 0.0                 0.0                 ;
        0.0                 0.0                 0.25189177424738    0.0                 0.0                 ;
        0.0                 0.0                 0.0                 0.54497475021237    0.0                 ;
        0.0                 0.0                 0.0                 0.08460416338212    0.22600748319395    ]

# Raymond J. Spiteri and Steven J. Ruuth. A new class of optimal high-order
# strong-stability-preserving time discretization methods. SIAM J. Numer. Anal.,
# 40(2):469–491, 2002.
StableRK4(step_size::Float64) = ExplicitRungeKuttaScheme(
        step_size,
        LinearAlgebra.LowerTriangular(_α),
        LinearAlgebra.LowerTriangular(_β)
    )

"""
Given `x0` apprximate `x0 exp(Dy)`.

    integrate_time(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
        y::Float64, scheme::ExplicitRungeKuttaScheme)

# Arguments
- `x0`: An initial row vector
- `D`: A square matrix
- `y`: time to integrate up to
- `h`: ExplicitRungeKuttaScheme.
"""
function integrate_time(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme)
    checksquare(D)
    !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
    return _integrate(x0,D,y,scheme)
end

function integrate_time(x0::SFMDistribution, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme)   
    return SFMDistribution(integrate_time(x0.coeffs,D,y,scheme),x0.dq)
end

function integrate_time(x0::SFMDistribution{DGMesh{T}}, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme; limiter::Limiter=GeneralisedMUSCL) where T
    checksquare(D)
    !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
    limiter_params = limiter.generate_params(x0.dq)
    limiter_function = x->limiter.fun(x,limiter_params...)
    return SFMDistribution(_integrate(x0.coeffs,D,y,scheme,limiter_function),x0.dq)
end

"""
    _integrate(x0::Array{Float64,2}, 
        D::Union{Array{Float64,2},SparseArrays.SparseMatrixCSC{Float64,Int}}, 
        y::Float64, scheme::ExplicitRungeKuttaScheme)

Use ExplicitRungeKuttaScheme method.
"""
function _integrate(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme)

    return _integrate(x0, D, y, scheme, identity)
end

function _integrate(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme, limit_function::Function)

    x = limit_function(x0[:])
    h = scheme.step_size
    l = size(scheme.alpha,1)
    α = scheme.alpha
    βh = scheme.beta*h
    v = Array{Float64,2}(undef,l+1,size(x0,2))
    vD = Array{Float64,2}(undef,l+1,size(x0,2))
    for t = h:h:y
        v[1,:] = x
        for i in 1:l
            vD[i,:] = transpose(v[i,:])*D
            initialised = false 
            for j in 1:i
                if α[i,j]!=0.0
                    initialised ? (v[i+1,:]+=v[j,:]*α[i,j]) : (v[i+1,:]=v[j,:]*α[i,j]; initialised=true)
                end
                if βh[i,j]!=0.0
                    initialised ? (v[i+1,:]+=vD[j,:]*βh[i,j]) : (v[i+1,:]=vD[j,:]*βh[i,j]; initialised=true)
                end
            end
            v[i+1,:] = limit_function(v[i+1,:])
        end
        x = v[end,:]
    end
    return Array(transpose(x))
end
