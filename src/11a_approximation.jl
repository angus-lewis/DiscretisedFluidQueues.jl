"""
    SFMDistribution(pdf::Function, dq::DiscretisedFluidQueue{<:Mesh}, fun_evals::Int = 6)

Return a discretised version of the `pdf`. The method of discretisation depends on the discretisation method 
of `dq`.

# Arguments:
- `pdf::Function`: a function `f(x::Float64,i::Int)` where `f(x,i)dx=P(X(0)∈dx,φ(0)=i)` is the initial 
    distribution of a fluid queue.
- `dqDiscretisedFluidQueue{<:Mesh}`: 
- `fun_evals`: the number of function evaluations used to approximate `f(x,i)`
"""
function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{<:Mesh})
end

"""
    SFMDistribution(pdf::Function, dq::DiscretisedFluidQueue{DGMesh{T}})

Approximates `pdf` via polynomials. 
"""
function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{DGMesh{T}}) where T
    cellnodes = cell_nodes(dq)
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            nodes = cellnodes[:,cell]
            if length(nodes)==1
                weights = Δ(dq,cell)
            else
                weights = gauss_lobatto_weights(nodes[1],nodes[end],length(nodes))
            end
            coeffs[:,cell,i] = pdf.(nodes,i).*weights
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

"""
    interior_point_mass(x::Float64, i::Int, dq::DiscretisedFluidQueue{<:Mesh})

Construct an approximation to a distribution of fluid queue with a point mass at (x,i). 
Returns a SFMDistribution. The method of approximation is defined by the type of <:Mesh.

# Arguments:
- `x::Float64`: The position of the point mass
- `i::Int`: The phase of the point mass
- `dq::DiscretisedFluidQueue`: 
"""
function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{<:Mesh}) end 
"""
    interior_point_mass(x::Float64, i::Int, dq::DiscretisedFluidQueue{DGMesh{T}})

Constructs a polynomial approximation to the point mass at (x,i)
"""
function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{DGMesh{T}}) where T
    (x<=dq.mesh.nodes[1])&&throw(DomainError("x is not in interior"))
    (x>=dq.mesh.nodes[end])&&throw(DomainError("x is not in interior"))
    if _has_right_boundary(dq.model.S,i) 
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,dq) 
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,dq) 
    end
    n₊ = N₊(dq)
    n₋ = N₋(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nodes = cell_nodes(dq.mesh)[:,cell_idx]
    V = vandermonde(n_bases_per_cell(dq))
    # we solve a∫ψ(x)ψ(x)'dx = ∫δ(x-x₀)ψ'(x)dx where ψ(x) is a column vector of 
    # lagrange polynomials on the current cell where each basis function integrates to 1.
    # The right-hand integral is ψ'(x₀). Let W = diag(V.w) and note W=W' and W^-1=(W^-1)'
    # Transform to a interpolating basis via ϕ(x) = (Δ/2×W)*ψ(x) where Δ/2V.w are the
    # integrals of the interpolating basis functions ϕ(x).
    # The inner product on the left-hand side 
    # becomes (2/Δ)W^-1 * ∫ϕ(x)ϕ(x)'dx * (2/Δ)W^-1. 
    # Maths tells us that ∫ϕ(x)ϕ(x)'dx = (V*V')^-1 Δ/2. 
    # Thus a∫ψ(x)ψ(x)'dx = ∫δ(x-x₀)ψ'(x)dx
    # becomes a(2/Δ)W^-1 * ∫ϕ(x)ϕ(x)'dx * (2/Δ)W^-1 = ϕ'(x₀)(2/Δ)W^-1
    # solving for a 
    # a = ϕ'(x₀) (2/Δ)W^-1 * ((2/Δ)W^-1*∫ϕ(x)ϕ(x)'dx*(2/Δ)W^-1)^-1
    #   = ϕ'(x₀) (∫ϕ(x)ϕ(x)'dx)^-1 * (Δ/2)W
    #   = ϕ'(x₀) (V.V*V.V')2/Δ * (Δ/2)W
    #   = ϕ'(x₀) (V.V*V.V') W
    coeffs[coeff_idx] = lagrange_polynomials(nodes,x)'*(V.V*V.V').*V.w'
    return SFMDistribution(coeffs,dq)
end

"""
    left_point_mass(i::Int, dq::DiscretisedFluidQueue)

Construct a SFMDistribution with a point mass at the left boundary of `dq` in phase `i` and 0 elsewhere.
"""
function left_point_mass(i::Int,dq::DiscretisedFluidQueue) 
    _has_right_boundary(dq.model.S,i)&&throw(DomainError("only phases with lpm=true have left point masses"))
    n₊ = N₊(dq)
    n₋ = N₊(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nᵢ = N₋(dq.model.S[1:i])
    coeffs[nᵢ] = 1.0
    return SFMDistribution(coeffs,dq)
end

"""
    right_point_mass(i::Int, dq::DiscretisedFluidQueue)

Construct a SFMDistribution with a point mass at the left boundary of `dq` in phase `i` and 0 elsewhere.
"""
function right_point_mass(i::Int,dq::DiscretisedFluidQueue)
    _has_left_boundary(dq.model.S,i)&&throw(DomainError("only phases with rpm=false have left point masses"))
    n₊ = N₊(dq)
    n₋ = N₊(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    nᵢ = N₊(dq.model.S[1:i])
    coeffs[end-n₊+nᵢ] = 1.0
    return SFMDistribution(coeffs,dq)
end

"""
    SFMDistribution(pdf::Function, dq::DiscretisedFluidQueue{FVMesh}, fun_evals::Int = 6)

Construct an approximation to `pdf` as the average of `pdf` of each cell. 

Uses quadrature with `fun_evals` function evaluations to approximate cell averages.
"""
function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{FVMesh{T}}, fun_evals::Int=6) where T
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            quad = gauss_lobatto_quadrature(x->pdf(x,i),a,b,fun_evals)
            coeffs[1,cell,i] = quad./Δ(dq,cell)
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

"""
    interior_point_mass(x::Float64, i::Int, dq::DiscretisedFluidQueue{FVMesh})

Construct an approximation to a point mass at (x,i). Basically, just puts mass 1 in the cell containing (x,i).
"""
function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{FVMesh{T}}) where T
    if _has_right_boundary(dq.model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_pos(x,i,dq) 
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_neg(x,i,dq) 
    end
    n₊ = N₊(dq)
    n₋ = N₋(dq)
    coeffs = zeros(1,n₊+n₋+total_n_bases(dq))
    coeffs[cell_idx] = 1.0./Δ(dq,cell_idx)
    return SFMDistribution(coeffs,dq)
end

## constructors for FRAPMesh
function SFMDistribution_from_cdf(cdf::Function,dq::DiscretisedFluidQueue{FRAPMesh{T}}; fun_evals::Int=10) where T
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            if _has_right_boundary(dq.model.S,i)
                o = expected_orbit_from_cdf(x->cdf(Δ(dq,cell)-x,i),a,b,fun_evals) 
            elseif _has_left_boundary(dq.model.S,i)
                o = expected_orbit_from_cdf(x->cdf(x,i),a,b,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq.model.S))]
    return SFMDistribution(coeffs,dq)
end

"""
    SFMDistribution(pdf::Function, dq::DiscretisedFluidQueue{FRAPMesh}; fun_evals = 100)

Return the appropriate initial condition to approximate the initial distribution `pdf`
for the numerical discretisation scheme defined by the `dq` DiscretisedFluidQueue.

i.e. for each cell compute the expected orbit position 
``∫pdf(x,i) ⋅ a(x) dx ``
where the integral is over each cell and 
``a(x) = a exp(S(x-yₖ))`` 
if the membership of `i` is `-1` and 
``a(x) = a exp(S(yₖ₊₁-x))`` 
if the membership of `i` is `1` where 
`yₖ` and `yₖ₊₁` are the left and right cell edges, respectively, and 
`a` and `S` are defined by `dq.mesh.me` are parameters of a MatrixExponential.
"""
function SFMDistribution(pdf::Function,dq::DiscretisedFluidQueue{FRAPMesh{T}}; fun_evals=100) where T
    n₋ = N₋(dq)
    coeffs = zeros(n_bases_per_cell(dq),n_intervals(dq),n_phases(dq))
    for i in phases(dq)
        for cell in 1:n_intervals(dq)
            a,b = dq.mesh.nodes[cell:cell+1]
            if _has_right_boundary(dq.model.S,i)
                o = expected_orbit_from_pdf(x->(b-a)*pdf(a+x*(b-a),i),dq.mesh.me,0.0,1.0,fun_evals) 
            else
                o = expected_orbit_from_pdf(x->(b-a)*pdf(b-x*(b-a),i),dq.mesh.me,0.0,1.0,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(dq))]
    return SFMDistribution(coeffs,dq)
end

"""
    interior_point_mass(x::Float64, i::Int, dq::DiscretisedFluidQueue{FRAPMesh})

Return the appropriate initial condition to approximate the initial distribution which 
is a point mass at (x,i) for the numerical discretisation scheme defined by the 
`dq` DiscretisedFluidQueue.

i.e. Compute the orbit position 
``a(x) = a exp(S(x-yₖ))`` 
if the membership of `i` is `-1` and 
``a(x) = a exp(S(yₖ₊₁-x))`` 
if the membership of `i` is `1` where 
`yₖ` and `yₖ₊₁` are the left and right cell edges and `x∈[yₖ,yₖ₊₁]`, respectively, and 
`a` and `S` are defined by `dq.mesh.me` are parameters of a MatrixExponential.
"""
function interior_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue{FRAPMesh{T}}) where T
    if _has_right_boundary(dq.model.S,i) 
        cell_idx, ~, coeff_idx = _get_coeff_index_pos(x,i,dq) 
        yₖ = dq.mesh.nodes[cell_idx]
        d = x-yₖ
    elseif _has_left_boundary(dq.model.S,i)
        cell_idx, ~, coeff_idx = _get_coeff_index_neg(x,i,dq)
        yₖ₊₁ = dq.mesh.nodes[cell_idx+1]
        d = yₖ₊₁-x
    end
    o = orbit(dq.mesh.me,d/Δ(dq.mesh,cell_idx))
    coeffs = zeros(1,N₊(dq)+N₋(dq)+total_n_bases(dq))
    coeffs[coeff_idx] = o
    return SFMDistribution(coeffs,dq)
end