function _error_on_nothing(idx)
    (idx===nothing) && throw(DomainError("x is not in the support of the mesh"))
end

function _get_nodes_coeffs_from_index(cell_idx::Int,i::Int,dq::DiscretisedFluidQueue) 
    cellnodes = cell_nodes(dq)[:,cell_idx]
    coeff_idx = (N₋(dq) + (i-1)*n_bases_per_phase(dq) + (cell_idx-1)*n_bases_per_cell(dq)) .+ (1:n_bases_per_cell(dq))
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
    coeff_idx = N₋(dq) + n_bases_per_phase(dq)*n_phases(dq) + N₊(dq.model.S[1:i])
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

function pdf(d::SFMDistribution{DGMesh})
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
                    coeffs = (2/(Δ(mesh)[cell_idx]))*(1.0./V.w).*coeffs
                end
                basis_values = lagrange_polynomials(cellnodes, x)
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

function pdf(d::SFMDistribution{FRAPMesh})
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.dq)
            coeffs = d.coeffs[coeff_idx]
            # if not a point mass, then reconstruct solution
            if _has_right_boundary(d.dq.model.S,i)
                yₖ₊₁ = mesh.nodes[cell_idx+1]
                to_go = yₖ₊₁-x
            elseif _has_left_boundary(d.dq.model.S,i)
                yₖ = mesh.nodes[cell_idx]
                to_go = x-yₖ
            end
            me = build_me(cme_params[n_bases_per_cell(mesh)], mean = Δ(mesh)[cell_idx])
            fxi = (pdf(Array(coeffs'),me,to_go) + pdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx]-to_go))./cdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx])
        end
        return fxi
    end
    return f
end

function pdf(d::SFMDistribution{FVMesh})
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.dq.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.dq)
            # if not a point mass, then reconstruct solution
            ptsLHS = Int(ceil(_order(mesh)/2))
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
            coeff_idx = (N₋(d.dq) + (i-1)*n_bases_per_phase(mesh)) .+ nodesIdx
            coeffs = d.coeffs[coeff_idx]#./Δ(mesh)[cell_idx]
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
            idx = (N₋(d.dq) + (i-1)*n_bases_per_phase(d.dq) + (cell-1)*n_bases_per_cell(d.dq)) .+ 1 
            c += d.coeffs[idx]
        end
    else
        for cell in 1:(cell_idx-1)
            idx = (N₋(d.dq) + (i-1)*n_bases_per_phase(d.dq) + (cell-1)*n_bases_per_cell(d.dq)) .+ (1:n_bases_per_cell(d.dq))
            c += sum(d.coeffs[idx])
        end
    end
    return c
end

function cdf(d::SFMDistribution{DGMesh})
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.dq.model)) && throw(DomainError("phase i must be in the support of the model"))
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
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,n_bases_per_cell(mesh))
                Fxi += quad
            end
            # add the RH point mass if  required
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
"""
    cdf(d::SFMDistribution, x, i) 

Evaluate cdf(d::SFMDistribution) at (x,i).
"""
cdf(d::SFMDistribution,x,i) = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
cdf(d::SFMDistribution,x::Float64,i::Int) = cdf(d)(x,i)
cdf(d::SFMDistribution,x::Int,i::Int) = cdf(d)(convert(Float64,x),i)

function cdf(d::SFMDistribution{FRAPMesh})
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
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,d.dq)
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
                        Fxi += mass*(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                elseif _has_left_boundary(d.dq.model.S,i)
                    yₖ = mesh.nodes[cell_idx]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (xd-yₖ)/Δ(mesh,cell_idx)
                        Fxi += mass*(1-(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0))#sum(a) - (ccdf(a,me,to_go) - ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
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

function _sum_cells_left(d::SFMDistribution{FVMesh}, i::Int, cell_idx::Int)
    c = 0
    for cell in 1:(cell_idx-1)
        # first legendre basis function =1 & has all the mass
        idx = N₋(d.dq) + (i-1)*n_bases_per_phase(d.dq) + cell
        c += d.coeffs[idx]*Δ(d.dq,cell)
    end
    return c
end

function cdf(d::SFMDistribution{FVMesh})
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
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,d.dq)
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