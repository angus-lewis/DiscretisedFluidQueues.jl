"""
    ForwardEuler(step_size::Float64)

Defines an Euler integration scheme to be used in `integrate_time`.

# Arguments 
-  `step_size::Float64`: the step size of the integration scheme.
"""
ForwardEuler(step_size::Float64) = 
    ExplicitRungeKuttaScheme(step_size,LinearAlgebra.LowerTriangular([0.0][:,:]),[0.0],[1.0])

_heuns_coeff_matrix = LinearAlgebra.LowerTriangular([0.0 0.0;
                                                     1.0 0.0])
Heuns(step_size::Float64) = 
    ExplicitRungeKuttaScheme(step_size,_heuns_coeff_matrix,[0.0;1.0],[0.5;0.5])

_ssprk3_coeff_matrix = LinearAlgebra.LowerTriangular([0.0 0.0 0.0;
                                                      1.0 0.0 0.0;
                                                      1/4 1/4 0.0])
StableRK3(step_size::Float64) = 
    ExplicitRungeKuttaScheme(step_size,_ssprk3_coeff_matrix,[0.0;1.0;1/2],[1/6;1/6;2/3])
                
alpha = [1.0                0.0                 0.0                 0.0                 0.0                 ;
         0.44437049406734   0.55562950593266    0.0                 0.0                 0.0                 ;
         0.62010185138540   0.0                 0.37989814861460    0.0                 0.0                 ;
         0.17807995410773   0.0                 0.0                 0.8219200458922     0.0                 ;
         0.00683325884039   0.0                 0.51723167208978    0.12759831133288    0.34833675773694    ]
beta = [0.39175222700392    0.0                 0.0                 0.0                 0.0                 ;
        0.0                 0.36841059262959    0.0                 0.0                 0.0                 ;
        0.0                 0.0                 0.25189177424738    0.0                 0.0                 ;
        0.0                 0.0                 0.0                 0.54497475021237    0.0                 ;
        0.0                 0.0                 0.0                 0.08460416338212    0.22600748319395    ]
# https://ntrs.nasa.gov/api/citations/19870013797/downloads/19870013797.pdf
# pg 159 (168) Nodal DG book
# Sec 1.1.2 https://personal.math.ubc.ca/~cbm/mscthesis/cbm-mscthesis.pdf
# _ssprk45_coeff_matrix = LinearAlgebra.LowerTriangular([0.0 0.0 0.0;
#     1.0 0.0 0.0;
#     1/4 1/4 0.0])
StableRK45(step_size::Float64) = 
    ExplicitRungeKuttaScheme(step_size,_ssprk3_coeff_matrix,[0.0;1.0;1/2],[1/6;1/6;2/3])

struct ExplicitRungeKuttaScheme
    step_size::Float64
    matrix::LinearAlgebra.LowerTriangular{Float64} # a
    nodes::Vector{Float64}           # c
    weights::Vector{Float64}         # b
    # butcher tableau 
    #   
    # c | a
    #   +---
    #     b
    function ExplicitRungeKuttaScheme(step_size::Float64,
        matrix::LinearAlgebra.LowerTriangular{Float64},
        nodes::Vector{Float64},weights::Vector{Float64})

        t1 = (step_size > 0)
        l1 = length(weights)
        l2 = length(nodes)
        l3, l4 = size(matrix)
        (!t1)&&throw(DomainError("step_size must be positive"))
        !(l1==l2)&&throw(DimensionMismatch("weights and nodes must have same length"))
        !(l1==l3==l4)&&throw(DimensionMismatch("matrix must be square and have size=length(nodes)"))
        return new(step_size,matrix,nodes,weights)
    end
end

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
    checksquare(D)
    !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
    return SFMDistribution(_integrate(x0.coeffs,D,y,scheme),x0.dq)
end
# function integrate_time(x0::SFMDistribution, D::FullGenerator, y::Float64, scheme::AbstractExplicitRungeKuttaScheme)
#     checksquare(D)
#     !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
#     return SFMDistribution(_integrate(x0.coeffs,D.B,y,scheme),x0.dq)
# end
# function integrate_time(x0::SFMDistribution, D::LazyGenerator, y::Float64, scheme::AbstractExplicitRungeKuttaScheme)
#     checksquare(D)
#     !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
#     return SFMDistribution(_integrate(x0.coeffs,D,y,scheme),x0.dq)
# end

# """

#     _integrate(x0::Array{Float64,2}, 
#         D::Union{Array{Float64,2},SparseArrays.SparseMatrixCSC{Float64,Int}}, 
#         y::Float64, scheme::RungeKutta4)

# Use RungeKutta4 method.
# """
# function _integrate(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
#     y::Float64, scheme::RungeKutta4)
#     x = x0
#     h = scheme.step_size
#     c1 = 1.0/6.0 
#     c2 = 6.0 + 3.0*h
#     D = D*h
#     for t = h:h:y
#         xD = x*D
#         xD² = xD*D
#         xD³ = xD²*D
#         dx = c1 * (c2*xD + xD² + xD³)
#         x = x + dx
#     end
#     return x
# end
# function _integrate(x0::Array{Float64,2}, D::LazyGenerator, 
#     y::Float64, scheme::RungeKutta4)
#     x = x0
#     h = scheme.step_size
#     c1 = 1.0/6.0 
#     c2 = 6.0 + 3.0*h
#     D = D*h
#     for t = h:h:y
#         xD = fast_mul(x,D)
#         xD² = fast_mul(xD,D)
#         xD³ = fast_mul(xD²,D)
#         dx = c1 * (c2*xD + xD² + xD³)
#         x = x + dx
#     end
#     return x
# end

# """
#     _integrate(x0::Array{Float64,2}, 
#         D::Union{Array{Float64,2},SparseArrays.SparseMatrixCSC{Float64,Int}}, 
#         y::Float64, scheme::Euler)

# Use Eulers method.
# """
# function _integrate(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
#     y::Float64, scheme::Euler)
#     x = x0
#     h = scheme.step_size
#     for t = h:h:y
#         dx = h * (x * D)
#         x = x + dx
#     end
#     return x
# end
# function _integrate(x0::Array{Float64,2}, D::LazyGenerator, 
#     y::Float64, scheme::Euler)
#     x = x0
#     h = scheme.step_size
#     for t = h:h:y
#         dx = h * (fast_mul(x, D))
#         x = x + dx
#     end
#     return x
# end

"""
    _integrate(x0::Array{Float64,2}, 
        D::Union{Array{Float64,2},SparseArrays.SparseMatrixCSC{Float64,Int}}, 
        y::Float64, scheme::ExplicitRungeKuttaScheme)

Use ExplicitRungeKuttaScheme method.
"""
function _integrate(x0::Array{Float64,2}, D::AbstractArray{Float64,2},
    y::Float64, scheme::ExplicitRungeKuttaScheme)

    x = x0
    h = scheme.step_size
    l = length(scheme.weights)
    matrix = scheme.matrix*h
    v = Array{Float64,2}(undef,l,size(x0,2))
    for t = h:h:y
        v[:,:] .= x
        v[1,:] = transpose(v[1,:])*D
        for i in 1:l-1
            for j in 1:i-1
                if matrix[i,j]!=0.0
                    v[i+1,:] += v[j,:]*matrix[i,j]
                end                
            end
            v[i+1,:] = transpose(v[i+1,:])*D
        end
        x += h * sum(scheme.weights.*v, dims=1)
    end
    return x
end
