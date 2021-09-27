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
    RungeKutta4 <: TimeIntegrationScheme

Defines an RungeKutta4 integration scheme to be used in `integrate_time`.

# Arguments 
-  `step_size::Float64`: the step size of the integration scheme.
"""
struct RungeKutta4 <: TimeIntegrationScheme
    step_size::Float64
end

"""

"""


"""
Given `x0` apprximate `x0 exp(Dy)`.

    integrate_time(x0::AbstractArray{Float64, 2}, D::Generator, y::Float64, scheme::TimeIntegrationScheme)

# Arguments
- `x0`: An initial row vector
- `D`: A square matrix
- `y`: time to integrate up to
- `h`: TimeIntegrationScheme.
"""
function integrate_time(x0::AbstractArray{Float64,2}, D::AbstractArray{Float64,2}, y::Float64, scheme::TimeIntegrationScheme)
    checksquare(D)
    !(size(x0,2)==size(D,1))&&throw(DimensionMismatch("x0 must have length size(D,1)"))
    
    return _integrate(x0,D,y,scheme)
end

"""
    integrate_time(x0::AbstractArray{Float64,2}, D::Generator, y::Float64, scheme::RungeKutta4)

Use RungeKutta4 method.
"""
function _integrate(x0::AbstractArray{Float64,2}, D::AbstractArray{Float64,2}, y::Float64, scheme::RungeKutta4)
    x = x0
    h = scheme.step_size
    c1 = 1.0/6.0 
    c2 = 6.0 + 3.0*h
    D = D*h
    for t = h:h:y
        xD = x*D
        xD² = xD*D
        xD³ = xD²*D
        dx = c1 * (c2*xD + xD² + xD³)
        x = x + dx
    end
    return x
end

"""
    integrate_time(x0::AbstractArray{Float64,2}, D::Generator, y::Float64, scheme::Euler)

Use Eulers method.
"""
function _integrate(x0::AbstractArray{Float64,2}, D::AbstractArray{Float64,2}, y::Float64, scheme::Euler)
    x = x0
    h = scheme.step_size
    for t = h:h:y
        dx = h * (x * D)
        x = x + dx
    end
    return x
end