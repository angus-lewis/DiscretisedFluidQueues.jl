abstract type TimeIntegrationScheme end

struct Euler <: TimeIntegrationScheme
    step_size::Float64
end

"""
Uses Eulers method to integrate the matrix DE ``f'(x) = f(x)D`` to
approxiamte ``f(y)``.

    EulerDG(
        D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
        y::Real,
        x0::Array{<:Real};
        h::Float64 = 0.0001,
    )

# Arguments
- `D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}}`:
    the matrix ``D`` in the system of ODEs ``f'(x) = f(x)D``.
- `y::Real`: the value where we want to evaluate ``f(y)``.
- `x0::Array{<:Real}`: a row-vector initial condition.
- `h::Float64`: a stepsize for theEuler scheme.

# Output
- `f(y)::Array`: a row-vector approximation to ``f(y)``
"""
function integrate_time(
    x0::AbstractArray{<:Real,2},
    D::Generator,
    y::Real,
    h::TimeIntegrationScheme = Euler(y./1000),
)
    x = x0
    for t = h:h:y
        dx = h * (x * D)
        x = x + dx
    end
    return x
end