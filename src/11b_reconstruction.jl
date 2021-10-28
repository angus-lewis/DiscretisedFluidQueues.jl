function _error_on_nothing(idx)
    (idx===nothing) && throw(DomainError("x is not in the support of the mesh"))
end

function _get_nodes_coeffs_from_index(cell_idx::Int,i::Int,dq::DiscretisedFluidQueue) 
    cellnodes = cell_nodes(dq)[:,cell_idx]
    coeff_idx = (N₋(dq) + (i-1)*n_bases_per_cell(dq) + (cell_idx-1)*n_bases_per_level(dq)) .+ (1:n_bases_per_cell(dq))
    return cellnodes, coeff_idx
end

function _get_coeff_index_pos(x::Float64,i::Int,dq::DiscretisedFluidQueue) 
    cell_idx = findlast(x.>=dq.mesh.nodes)
    _error_on_nothing(cell_idx)
    cellnodes, coeff_idx = _get_nodes_coeffs_from_index(cell_idx,i,dq)
    return cell_idx, cellnodes, coeff_idx
end
function _get_coeff_index_neg(x::Float64,i::Int,dq::DiscretisedFluidQueue) 
    cell_idx = findfirst(x.<=dq.mesh.nodes) - 1
    _error_on_nothing(cell_idx)
    cellnodes, coeff_idx = _get_nodes_coeffs_from_index(cell_idx,i,dq)
    return cell_idx, cellnodes, coeff_idx
end

function _get_point_mass_data_pos(i::Int,dq)
    cellnodes = dq.mesh.nodes[end]
    coeff_idx = N₋(dq) + total_n_bases(dq) + N₊(dq.model.S[1:i])
    return cellnodes, coeff_idx 
end
function _get_point_mass_data_neg(i::Int,dq::DiscretisedFluidQueue)
    cellnodes = dq.mesh.nodes[1]
    coeff_idx = N₋(dq.model.S[1:i])
    return cellnodes, coeff_idx 
end
_is_left_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue) = 
    (x==dq.mesh.nodes[1])&&_has_left_boundary(dq.model.S,i)
_is_right_point_mass(x::Float64,i::Int,dq::DiscretisedFluidQueue) = 
    (x==dq.mesh.nodes[end])&&_has_right_boundary(dq.model.S,i)
    

function _get_coeffs_index(x::Float64,i::Int,dq::DiscretisedFluidQueue)
    !(i∈phases(dq)) && throw(DomainError("phase i must be in the support of the model"))

    # find which inteval Dₗ,ᵢ x is in 
    if _is_left_point_mass(x,i,dq)
        cell_idx = "point mass"
        cellnodes, coeff_idx = _get_point_mass_data_neg(i,dq)
    elseif _is_right_point_mass(x,i,dq)
        cell_idx = "point mass"
        cellnodes, coeff_idx = _get_point_mass_data_pos(i,dq)
    else # not a point mass 
        if _has_right_boundary(dq.model.S,i)
            cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,dq) 
        elseif _has_left_boundary(dq.model.S,i)
            cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,dq) 
        end
    end

    return cell_idx, cellnodes, coeff_idx
end

"""
    pdf(d::SFMDistribution)

Return a function of two variables (x::Float64,i::Int) which is the 
probability distribution function defined by `d`. Approximates the density function of 
a fluid queue.
"""
function pdf(d::SFMDistribution) 
    throw(DomainError("unknown <:Mesh}"))
end

function pdf(d::SFMDistribution{DGMesh{T}}) where T
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            #fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.dq)
            coeffs = d.coeffs[coeff_idx]
            # if not a point mass, then reconstruct solution
            # if !(cell_idx=="point mass")
                if basis(mesh) == "legendre"
                    coeffs = legendre_to_lagrange(coeffs)
                else
                    V = vandermonde(n_bases_per_cell(mesh))
                    coeffs = coeffs
                end
                lp = lagrange_polynomials(cellnodes, x) # interpolating basis
                # transform to lagrange basis where each basis function integrates to 1,
                # i.e. divide the basis functions by their integrals.
                # their integrals are V.w*Δ/2
                basis_values = (lp./V.w)*2/Δ(d.dq,cell_idx) 
                fxi = LinearAlgebra.dot(basis_values,coeffs)
            # else 
            #     fxi = coeffs
            # end
        end
        return fxi
    end
    return f
end
"""
    pdf(d::SFMDistribution, x, i) 

Evaluate pdf(d::SFMDistribution) at (x,i).
"""
pdf(d::SFMDistribution,x,i) = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
pdf(d::SFMDistribution,x::Float64,i::Int) = pdf(d)(x,i)
pdf(d::SFMDistribution,x::Int,i::Int) = pdf(d)(convert(Float64,x),i)

# abstract type ClosingOperator end

# struct UnnormalisedClosingOperator <: ClosingOperator 
#     pdf::Function
#     cdf::Function
# end
# struct NaiveNormalisedClosingOperator <: ClosingOperator 
#     cdf::Function
# end
# struct NormalisedClosingOperator <: ClosingOperator end

function unnormalised_closing_operator_pdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential) 
    #
    return x->pdf(a,me,x)
end
function unnormalised_closing_operator_cdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential) 
    return x->ccdf(a,me,x)
end

function naive_normalised_closing_operator_pdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential) 
    #
    return x->(pdf(a,me,x) + pdf(a,me,2.0-x))./cdf(a,me,2.0)
end
function naive_normalised_closing_operator_cdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential)
    #
    return x->(ccdf(a,me,x)-ccdf(a,me,2.0-x))/cdf(a,me,2.0)
end

function normalised_closing_operator_pdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential) 
    # a [exp(Sx) + exp(S(2m-x))][I-exp(S2m)]^-1 s
    tmp = LinearAlgebra.I(size(me.S,1)) - exp(me.S*2.0)
    inv_factor = inv(tmp)

    return x->only(a*(exp(me.S*x)+exp(me.S*(2.0-x)))*inv_factor*me.s)
end
function normalised_closing_operator_cdf(a::AbstractArray{Float64,2},
    me::AbstractMatrixExponential)
    # 
    tmp = LinearAlgebra.I(size(me.S,1)) - exp(me.S*2.0)
    inv_factor = inv(tmp)

    exp_factor(x) = LinearAlgebra.I(size(me.S,1))-(exp(me.S*x)-exp(me.S*(2.0-x))+exp(me.S*2.0))
    return x->only(1.0.-sum(a*exp_factor(x)*inv_factor,dims=2))
end

function pdf(d::SFMDistribution{FRAPMesh{T}}, 
    closing_operator::Function=normalised_closing_operator_pdf) where T

    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, ~, coeff_idx = _get_coeffs_index(x,i,d.dq)
            # if not a point mass, then reconstruct solution
            if _has_right_boundary(d.dq.model.S,i)
                yₖ₊₁ = mesh.nodes[cell_idx+1]
                to_go = (yₖ₊₁-x)./Δ(mesh,cell_idx)
            elseif _has_left_boundary(d.dq.model.S,i)
                yₖ = mesh.nodes[cell_idx]
                to_go = (x-yₖ)./Δ(mesh,cell_idx)
            end
            me = mesh.me
            fxi = closing_operator(Array(transpose(d.coeffs[coeff_idx])),me)(to_go)./Δ(mesh,cell_idx)
        end
        return fxi
    end
    return f
end

function pdf(d::SFMDistribution{FVMesh{T}}) where T
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, ~, coeff_idx = _get_coeffs_index(x,i,d.dq)
            # if not a point mass, then reconstruct solution
            ptsLHS = (membership(d.dq.model.S,i)<0) ? (Int(ceil(_order(mesh)/2))+1) : Int(ceil(_order(mesh)/2))
            if cell_idx-ptsLHS < 0
                nodesIdx = 1:_order(mesh)
                nodes = cell_nodes(mesh)[nodesIdx]
                poly_vals = lagrange_polynomials(nodes,x)
            elseif cell_idx-ptsLHS+_order(mesh) > n_bases_per_phase(mesh)
                nodesIdx = (n_bases_per_phase(mesh)-_order(mesh)+1):n_bases_per_phase(mesh)
                nodes = cell_nodes(mesh)[nodesIdx]
                poly_vals = lagrange_polynomials(nodes,x)
            else
                nodesIdx =  (cell_idx-ptsLHS) .+ (1:_order(mesh))
                poly_vals = lagrange_polynomials(cell_nodes(mesh)[nodesIdx],x)
            end
            coeff_idx = (N₋(d.dq) + i) .+ ((nodesIdx.-1).*n_phases(d.dq))
            coeffs = d.coeffs[coeff_idx]
            fxi = LinearAlgebra.dot(poly_vals,coeffs)
        end
        return fxi
    end
    return f
end

############
### CDFs ###
############
"""
    cdf(d::SFMDistribution)

Return a function of two variables (x::Float64,i::Int) which is the 
cumulative distribution distribution function defined by `d`. Approximates the distribution function of 
a fluid queue.
"""
function cdf(d::SFMDistribution) 
    throw(DomainError("unknown <:Mesh"))
end

"""

    _sum_cells_left(d::SFMDistribution, i::Int, cell_idx::Int, dq::DiscretisedFluidQueue)

Add up all the probability mass in phase `i` in the cells to the left of `cell_idx`.
"""
function _sum_cells_left(d::SFMDistribution, i::Int, cell_idx::Int) 
    c = 0.0
    if basis(d.dq.mesh) == "legendre"
        for cell in 1:(cell_idx-1)
            # first legendre basis function =1 & has all the mass
            idx = (N₋(d.dq) + (i-1)*n_bases_per_cell(d.dq) + (cell-1)*n_bases_per_level(d.dq)) .+ 1 
            c += d.coeffs[idx]
        end
    else
        for cell in 1:(cell_idx-1)
            idx = (N₋(d.dq) + (i-1)*n_bases_per_cell(d.dq) + (cell-1)*n_bases_per_level(d.dq)) .+ (1:n_bases_per_cell(d.dq))
            c += sum(d.coeffs[idx])
        end
    end
    return c
end

function cdf(d::SFMDistribution{DGMesh{T}}) where T
    # First, get the coeffs and project in to higher dimensional space
    # get coeffs without boundary masses
    coeffs = d.coeffs[N₋(d.dq)+1:end-N₊(d.dq)] 
    coeffs = reshape(coeffs,n_bases_per_cell(d.dq),n_phases(d.dq),n_intervals(d.dq))
    # reweight basis to get it into the typicall lagrange interpolating basis
    v = vandermonde(n_bases_per_cell(d.dq))
    for i in 1:n_phases(d.dq)
        coeffs[:,i,:] = (coeffs[:,i,:] ./ Δ(d.dq)') ./ (v.w ./ 2.0)
    end
    # coeffs = lagrange_to_legendre(coeffs) # do this later to save computation power
    # project to higher space (in lagrange this is equiv. to adding a 0 coeff)
    # coeffs = cat(coeffs,zeros(1,n_intervals(d.dq),n_phases(d.dq)),dims=1)# do this later
    V = vandermonde(n_bases_per_cell(d.dq)+1)
    # differentiation operator in legendre basis of degree n+1
    # i.e. ϕ(x) a legendre basis for polynomials of degree n+1
    # p(x) = ϕ(x)a, where a is a vector of coeffs, a polynomial
    # p'(x) = ϕ'(x)a = ϕ(x)Ga
    G = V.inv*V.D 
    # We have p'(x) = ϕ(x)a, we want to recover p(x).
    # p(x) is a polynomial of order n+1 so has a representation ϕ(x)b
    # and therefore p'(x)=ϕ'(x)b. To recover p(x) we want to find b.
    # We have, p'(x) = ϕ'(x)b = ϕ(x)a
    # multiplty by ϕᵀ(x) and integrate 
    # ⟹ ∫ϕᵀ(x)ϕ'(x)b dx = ∫ϕᵀ(x)ϕ(x)a dx
    # ⟹ Gb=Ma=a (as M=I)
    # but G is not invertible (which makes sense as integrals are only
    # known up to an additive constant). If we have the additional condition that 
    # p(-1)=0, then we can invert. i.e. ϕ(-1)b = 0.
    # First, evaluate ϕ(-1)
    ϕ₋₁ = V.V[1,:] 
    # add condition to G
    G[end,:] .= ϕ₋₁
    # now G is invertible. We also need to add the constraint to the right-hand side 
    # of the system. The system is 
    # G⋅b = coeffs, with the additional constraint begin G[end,:]⋅b = 0, i.e. 
    # we should set coeffs[:,end,:].=0, but this is already the case (the cat() command above) 
    # coeffs[:,end,:] .= 0.0
    # 
    # now solve it! 
    # Ginv = G\I # = G^-1
    # also, transform back to lagrange basis for easy evaluation, 
    # In summary multiply by V(n+1) G⁻¹(n+1) [I(n); zeros(n)] V⁻¹(n)
    # where the order of each operator is in ()
    # precompute transform
    transform = (V.V/G)*[LinearAlgebra.I(n_bases_per_cell(d.dq)); zeros(1,n_bases_per_cell(d.dq))]*v.inv
    integral_coeffs = zeros(n_bases_per_cell(d.dq)+1,n_phases(d.dq),n_intervals(d.dq))
    for i in 1:n_phases(d.dq)
        integral_coeffs[:,i,:] = (transform*coeffs[:,i,:]).*Δ(d.dq)'./2.0 # the factor at the end
        # is actually part of the integral operator G^-1. i.e. we should have Δ(d.dq,cell)G^-1 
        # for each cell. 
    end
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq.model)) && throw(DomainError("phase i must be in the support of the model"))
        mesh = d.dq.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else # x>= mesh.nodes[1]
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(d.dq.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,d.dq)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            if (x>mesh.nodes[1])&&(x<mesh.nodes[end])
                cell_idx, ~, ~ = _get_coeffs_index(x,i,d.dq)
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx)

                # integrate up to x in the cell which contains x
                # i.e. evaluate p(x) from above
                cellnodes = gauss_lobatto_points(mesh.nodes[cell_idx],
                    mesh.nodes[cell_idx+1], n_bases_per_cell(d.dq)+1)
                    
                basis_values = lagrange_polynomials(cellnodes, x) # interpolating basis
                Fxi += LinearAlgebra.dot(basis_values,integral_coeffs[:,i,cell_idx])
            elseif x>=mesh.nodes[end] # integrate over the whole space
                Fxi += _sum_cells_left(d, i, n_intervals(d.dq)+1)
                if _has_right_boundary(d.dq.model.S,i)
                    ~, right_pm_idx = _get_point_mass_data_pos(i,d.dq)
                    right_pm = d.coeffs[right_pm_idx]
                    Fxi += right_pm
                end
            end
        end
        return Fxi
    end
    return F
end
"""
    cdf(d::SFMDistribution, x, i) 

Evaluate cdf(d::SFMDistribution) at (x,i).
"""
cdf(d::SFMDistribution,x,i) = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
cdf(d::SFMDistribution,x::Float64,i::Int) = cdf(d)(x,i)
cdf(d::SFMDistribution,x::Int,i::Int) = cdf(d)(convert(Float64,x),i)

function cdf(d::SFMDistribution{FRAPMesh{T}}, 
    closing_operator::Function=normalised_closing_operator_cdf) where T

    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if _has_left_boundary(d.dq.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,d.dq)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, ~, coeff_idx = _get_coeffs_index(xd,i,d.dq)
            coeffs = d.coeffs[coeff_idx]

            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx)
                
                # integrate up to x in the cell which contains x
                # me = build_me(cme_params[n_bases_per_cell(mesh)], mean = Δ(mesh)[cell_idx])
                me = mesh.me
                a = Array(coeffs')
                if _has_right_boundary(d.dq.model.S,i)
                    yₖ₊₁ = mesh.nodes[cell_idx+1]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (yₖ₊₁-xd)/Δ(mesh,cell_idx)
                        Fxi += mass*closing_operator(a,me)(to_go)
                    end
                elseif _has_left_boundary(d.dq.model.S,i)
                    yₖ = mesh.nodes[cell_idx]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (xd-yₖ)/Δ(mesh,cell_idx)
                        Fxi += mass*(1-closing_operator(a,me)(to_go)) # sum(a) - (ccdf(a,me,to_go) - ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                end
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(d.dq.model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,d.dq)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end

function _sum_cells_left(d::SFMDistribution{FVMesh{T}}, i::Int, cell_idx::Int) where T
    c = 0
    for cell in 1:(cell_idx-1)
        idx = N₋(d.dq) + i + (cell-1)*n_bases_per_level(d.dq)
        c += d.coeffs[idx]*Δ(d.dq,cell)
    end
    return c
end

function cdf(d::SFMDistribution{FVMesh{T}}) where T
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(d.dq.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,d.dq)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, ~, ~ = _get_coeffs_index(xd,i,d.dq)
            # if not a point mass, then reconstruct solution
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,_order(mesh))
                Fxi += quad 
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(d.dq.model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,d.dq)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end

function cell_probs(d::SFMDistribution)
    function p(x::Float64,i::Int)
        _x_in_bounds = (x>d.dq.mesh.nodes[1])&&(x<d.dq.mesh.nodes[end])
        if _x_in_bounds
            ~, ~, coeff_idx = _get_coeff_index_pos(x,i,d.dq)
            return sum(d.coeffs[coeff_idx])
        else 
            return 0.0
        end
    end
    return p
end

function cell_probs(d::SFMDistribution{FVMesh{T}}) where T
    function p(x::Float64,i::Int)
        _x_in_bounds = (x>d.dq.mesh.nodes[1])&&(x<d.dq.mesh.nodes[end])
        if _x_in_bounds
            cell_idx, ~, coeff_idx = _get_coeff_index_pos(x,i,d.dq)
            cell_average = d.coeffs[coeff_idx]
            return only(cell_average*Δ(d.dq,cell_idx))
        else 
            return 0.0
        end
    end
    return p
end