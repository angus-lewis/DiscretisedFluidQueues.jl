function _error_on_nothing(idx)
    (idx===nothing) && throw(DomainError("x is not in the support of the mesh"))
end

function _get_nodes_coeffs_from_index(cell_idx::Int,i::Int,mesh::Mesh,model::Model) 
    cellnodes = cell_nodes(mesh)[:,cell_idx]
    coeff_idx = (N₋(model.S) + (i-1)*total_n_bases(mesh) + (cell_idx-1)*n_bases(mesh)) .+ (1:n_bases(mesh))
    return cellnodes, coeff_idx
end

function _get_coeff_index_pos(x::Float64,i::Int,mesh::Mesh,model::Model) 
    cell_idx = findlast(x.>=mesh.nodes)
    _error_on_nothing(cell_idx)
    cellnodes, coeff_idx = _get_nodes_coeffs_from_index(cell_idx,i,mesh,model)
    return cell_idx, cellnodes, coeff_idx
end
function _get_coeff_index_neg(x::Float64,i::Int,mesh::Mesh,model::Model) 
    cell_idx = findfirst(x.<=mesh.nodes) - 1
    _error_on_nothing(cell_idx)
    cellnodes, coeff_idx = _get_nodes_coeffs_from_index(cell_idx,i,mesh,model)
    return cell_idx, cellnodes, coeff_idx
end

function _get_point_mass_data_pos(i::Int,mesh::Mesh,model::Model)
    cellnodes = mesh.nodes[end]
    coeff_idx = N₋(model.S) + total_n_bases(mesh)*n_phases(model) + N₊(model.S[1:i])
    return cellnodes, coeff_idx 
end
function _get_point_mass_data_neg(i::Int,mesh::Mesh,model::Model)
    cellnodes = mesh.nodes[1]
    coeff_idx = N₋(model.S[1:i])
    return cellnodes, coeff_idx 
end
_is_left_point_mass(x::Float64,i::Int,mesh::Mesh,model::Model) = 
    (x==mesh.nodes[1])&&_has_left_boundary(model.S,i)
_is_right_point_mass(x::Float64,i::Int,mesh::Mesh,model::Model) = 
    (x==mesh.nodes[end])&&_has_right_boundary(model.S,i)
    

function _get_coeffs_index(x::Float64,i::Int,model::Model,mesh::Mesh)
    !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))

    # find which inteval Dₗ,ᵢ x is in 
    if _is_left_point_mass(x,i,mesh,model)
        cell_idx = "point mass"
        cellnodes, coeff_idx = _get_point_mass_data_neg(i,mesh,model)
    elseif _is_right_point_mass(x,i,mesh,model)
        cell_idx = "point mass"
        cellnodes, coeff_idx = _get_point_mass_data_pos(i,mesh,model)
    else # not a point mass 
        if _has_right_boundary(model.S,i)
            cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,mesh,model) 
        elseif _has_left_boundary(model.S,i)
            cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,mesh,model) 
        end
    end

    return cell_idx, cellnodes, coeff_idx
end

function pdf(d::SFMDistribution{T}) where T<:Mesh
    throw(DomainError("unknown SFMDistribution{<:Mesh}"))
end

function pdf(d::SFMDistribution{DGMesh})
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            #fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.model,mesh)
            coeffs = d.coeffs[coeff_idx]
            # if not a point mass, then reconstruct solution
            # if !(cell_idx=="point mass")
                if basis(mesh) == "legendre"
                    coeffs = legendre_to_lagrange(coeffs)
                else
                    V = vandermonde(n_bases(mesh))
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
pdf(d::SFMDistribution{T},x,i) where T<:Mesh = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
pdf(d::SFMDistribution{T},x::Float64,i::Int) where T<:Mesh = pdf(d,model)(x,i)
pdf(d::SFMDistribution{T},x::Int,i::Int) where T<:Mesh = pdf(d,model)(convert(Float64,x),i)

function pdf(d::SFMDistribution{FRAPMesh})
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.model,mesh)
            coeffs = d.coeffs[coeff_idx]
            # if not a point mass, then reconstruct solution
            if _has_right_boundary(d.model.S,i)
                yₖ₊₁ = mesh.nodes[cell_idx+1]
                to_go = yₖ₊₁-x
            elseif _has_left_boundary(d.model.S,i)
                yₖ = mesh.nodes[cell_idx]
                to_go = x-yₖ
            end
            me = MakeME(CMEParams[n_bases(mesh)], mean = Δ(mesh)[cell_idx])
            fxi = (pdf(Array(coeffs'),me,to_go) + pdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx]-to_go))./cdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx])
        end
        return fxi
    end
    return f
end

function pdf(d::SFMDistribution{FVMesh})
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,d.model,mesh)
            # if not a point mass, then reconstruct solution
            ptsLHS = Int(ceil(_order(mesh)/2))
            if cell_idx-ptsLHS < 0
                nodesIdx = 1:_order(mesh)
                nodes = cell_nodes(mesh)[nodesIdx]
                poly_vals = lagrange_polynomials(nodes,x)
            elseif cell_idx-ptsLHS+_order(mesh) > total_n_bases(mesh)
                nodesIdx = (total_n_bases(mesh)-_order(mesh)+1):total_n_bases(mesh)
                nodes = cell_nodes(mesh)[nodesIdx]
                poly_vals = lagrange_polynomials(nodes,x)
            else
                nodesIdx =  (cell_idx-ptsLHS) .+ (1:_order(mesh))
                poly_vals = lagrange_polynomials(cell_nodes(mesh)[nodesIdx],x)
            end
            coeff_idx = (N₋(d.model.S) + (i-1)*total_n_bases(mesh)) .+ nodesIdx
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

function cdf(d::SFMDistribution{T}) where T<:Mesh
    throw(DomainError("unknown SFMDistribution{<:Mesh}"))
end

"""

    _sum_cells_left(d::SFMDistribution, i::Int, cell_idx::Int, mesh::Mesh, model::Model)

Add up all the probability mass in phase `i` in the cells to the left of `cell_idx`.
"""
function _sum_cells_left(d::SFMDistribution, i::Int, cell_idx::Int, mesh::Mesh, model::Model)
    c = 0.0
    if basis(mesh) == "legendre"
        for cell in 1:(cell_idx-1)
            # first legendre basis function =1 & has all the mass
            idx = (N₋(d.model.S) + (i-1)*total_n_bases(mesh) + (cell-1)*n_bases(mesh)) .+ 1 
            c += d.coeffs[idx]
        end
    else
        for cell in 1:(cell_idx-1)
            idx = (N₋(d.model.S) + (i-1)*total_n_bases(mesh) + (cell-1)*n_bases(mesh)) .+ (1:n_bases(mesh))
            c += sum(d.coeffs[idx])
        end
    end
    return c
end

function cdf(d::SFMDistribution{DGMesh})
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(d.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,d.model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, ~, ~ = _get_coeffs_index(xd,i,d.model,mesh)
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx, mesh, d.model)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,n_bases(mesh))
                Fxi += quad
            end
            # add the RH point mass if  required
            if (x>=mesh.nodes[end])&&_has_right_boundary(d.model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,d.model)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end
cdf(d::SFMDistribution{T},x,i) where T<:Mesh = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
cdf(d::SFMDistribution{T},x::Float64,i::Int) where T<:Mesh = cdf(d)(x,i)
cdf(d::SFMDistribution{T},x::Int,i::Int) where T<:Mesh = cdf(d)(convert(Float64,x),i)

function cdf(d::SFMDistribution{FRAPMesh})
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if _has_left_boundary(d.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,d.model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,d.model,mesh)
            coeffs = d.coeffs[coeff_idx]

            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx, mesh, d.model)
                
                # integrate up to x in the cell which contains x
                # me = MakeME(CMEParams[n_bases(mesh)], mean = Δ(mesh)[cell_idx])
                me = mesh.me
                a = Array(coeffs')
                if _has_right_boundary(d.model.S,i)
                    yₖ₊₁ = mesh.nodes[cell_idx+1]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (yₖ₊₁-xd)/Δ(mesh,cell_idx)
                        Fxi += mass*(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                elseif _has_left_boundary(d.model.S,i)
                    yₖ = mesh.nodes[cell_idx]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (xd-yₖ)/Δ(mesh,cell_idx)
                        Fxi += mass*(1-(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0))#sum(a) - (ccdf(a,me,to_go) - ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                end
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(d.model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,d.model)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end

function _sum_cells_left_fv(d, i, cell_idx, mesh, model)
    c = 0
    for cell in 1:(cell_idx-1)
        # first legendre basis function =1 & has all the mass
        idx = N₋(d.model.S) + (i-1)*total_n_bases(mesh) + cell
        c += d.coeffs[idx]*Δ(mesh)[cell]
    end
    return c
end

function cdf(d::SFMDistribution{FVMesh})
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(d.model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(d.model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,d.model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,d.model,mesh)
            # if not a point mass, then reconstruct solution
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left_fv(d, i, cell_idx, mesh, d.model)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,_order(mesh))
                Fxi += quad 
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(d.model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,d.model)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end