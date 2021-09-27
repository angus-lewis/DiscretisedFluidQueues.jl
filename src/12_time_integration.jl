"""
    TimeIntegrationScheme
"""
abstract type TimeIntegrationScheme end

"""
    Euler <: TimeIntegrationScheme

Defines an Euler integration scheme to be used in `integrate_time`.

# Arguments 
-  `step_size::Float64`: the step size of the integration scheme.
"""
struct Euler <: TimeIntegrationScheme
    step_size::Float64
end

"""
Given `x0` apprximate `x0 exp(Dy)`.

    integrate_time(x0::AbstractArray{<:Real, 2}, D::Generator, y::Real, h::TimeIntegrationScheme = Euler(y ./ 1000))

# Arguments
- `x0`: An initial row vector
- `D`: A square matrix
- `y`: time to integrate up to
- `h`: TimeIntegrationScheme.

# Output
- 
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
