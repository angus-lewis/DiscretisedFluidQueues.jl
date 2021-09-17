# import Base: getindex, size, *

struct SFMDistribution{T<:Mesh} <: AbstractArray{Float64,2} 
    coeffs::Array{Float64,2}
    model::Model
    mesh::T
    SFMDistribution{T}(coeffs::Array{Float64,2},model::Model,mesh::T) where T<:Mesh = 
        (size(coeffs,1)==1) ? new(coeffs,model,mesh) : throw(DimensionMismatch("coeffs must be a row-vector"))
end

SFMDistribution(coeffs::Array{Float64,2},model::Model,mesh::T) where T = 
    SFMDistribution{T}(coeffs,model,mesh)

SFMDistribution(model::Model,mesh::T) where T = 
    SFMDistribution{T}(zeros(1,total_n_bases(mesh)*n_phases(mesh)+N₊(model.S))+N₋(model.S))

function SFMDistribution(pdf::Function,model::Model,mesh::DGMesh)
    cellnodes = cell_nodes(mesh)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            nodes = cellnodes[:,cell]
            weights = gauss_lobatto_weights(nodes[1],nodes[end],length(nodes))
            coeffs[:,cell,i] = pdf.(nodes,i).*weights
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::DGMesh)
    (x<=mesh.nodes[1])&&throw(DomainError("x is not in interior"))
    (x>=mesh.nodes[end])&&throw(DomainError("x is not in interior"))
    if _has_right_boundary(model.S,i) 
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_pos(x,i,mesh,model) 
    elseif _has_left_boundary(model.S,i)
        cell_idx, cellnodes, coeff_idx = _get_coeff_index_neg(x,i,mesh,model) 
    end
    n₊ = N₊(model.S)
    n₋ = N₋(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nodes = cell_nodes(mesh)[:,cell_idx]
    coeffs[coeff_idx] = lagrange_polynomials(nodes,x)
    return SFMDistribution(coeffs,model,mesh)
end

function left_point_mass(i::Int,model::Model,mesh::Mesh)
    _has_right_boundary(model.S,i)&&throw(DomainError("only phases with membership = -1.0 or -0.0 have left point masses"))
    n₊ = N₊(model.S)
    n₋ = N₊(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nᵢ = N₋(model.S[1:i])
    coeffs[nᵢ] = 1.0
    return SFMDistribution(coeffs,model,mesh)
end

function right_point_mass(i::Int,model::Model,mesh::Mesh)
    _has_left_boundary(model.S,i)&&throw(DomainError("only phases with membership = 1.0 or 0.0 have left point masses"))
    n₊ = N₊(model.S)
    n₋ = N₊(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    nᵢ = N₊(model.S[1:i])
    coeffs[end-n₊+nᵢ] = 1.0
    return SFMDistribution(coeffs,model,mesh)
end

function SFMDistribution(pdf::Function,model::Model,mesh::FVMesh, fun_evals::Int=6)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            quad = gauss_lobatto_quadrature(x->pdf(x,i),a,b,fun_evals)
            coeffs[1,cell,i] = quad./Δ(mesh,cell)
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::FVMesh)
    if _has_right_boundary(model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_pos(x,i,mesh,model) 
    elseif _has_left_boundary(model.S,i)
        cell_idx, ~, ~ = _get_coeff_index_neg(x,i,mesh,model) 
    end
    n₊ = N₊(model.S)
    n₋ = N₋(model.S)
    coeffs = zeros(1,n₊+n₋+total_n_bases(mesh)*n_phases(model))
    coeffs[cell_idx] = 1.0./Δ(mesh,cell_idx)
    return SFMDistribution(coeffs,model,mesh)
end

## constructors for FRAPMesh
function SFMDistribution_from_cdf(cdf::Function,model::Model,mesh::FRAPMesh; fun_evals::Int=10)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            if _has_right_boundary(model.S,i)
                o = expected_orbit_from_cdf(x->cdf(Δ(mesh,cell)-x,i),a,b,fun_evals) 
            elseif _has_left_boundary(model.S,i)
                o = expected_orbit_from_cdf(x->cdf(x,i),a,b,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function SFMDistribution(pdf::Function,model::Model,mesh::FRAPMesh; fun_evals=100)
    n₋ = N₋(model.S)
    coeffs = zeros(n_bases(mesh),n_intervals(mesh),n_phases(model))
    for i in phases(model)
        for cell in 1:n_intervals(mesh)
            a,b = mesh.nodes[cell:cell+1]
            if _has_right_boundary(model.S,i)
                o = expected_orbit_from_pdf(x->(b-a)*pdf(a+x*(b-a),i),mesh.me,0.0,1.0,fun_evals) 
            else
                o = expected_orbit_from_pdf(x->(b-a)*pdf(b-x*(b-a),i),mesh.me,0.0,1.0,fun_evals) 
            end
            coeffs[:,cell,i] = o
        end
    end
    coeffs = [zeros(1,n₋) Array(coeffs[:]') zeros(1,N₊(model.S))]
    return SFMDistribution(coeffs,model,mesh)
end

function interior_point_mass(x::Float64,i::Int,model::Model,mesh::FRAPMesh)
    if _has_right_boundary(model.S,i) 
        cell_idx, ~, coeff_idx = _get_coeff_index_pos(x,i,mesh,model) 
        yₖ = mesh.nodes[cell_idx]
        d = x-yₖ
    elseif _has_left_boundary(model.S,i)
        cell_idx, ~, coeff_idx = _get_coeff_index_neg(x,i,mesh,model)
        yₖ₊₁ = mesh.nodes[cell_idx+1]
        d = yₖ₊₁-x
    end
    o = orbit(mesh.me,d/Δ(mesh,cell_idx))
    coeffs = zeros(1,N₊(model.S)+N₋(model.S)+total_n_bases(mesh)*n_phases(model))
    coeffs[coeff_idx] = o
    return SFMDistribution(coeffs,model,mesh)
end

+(f::SFMDistribution,g::SFMDistribution) = throw(DomainError("cannot add SFMDistributions with differen mesh types"))
function +(f::SFMDistribution{T},g::SFMDistribution{T}) where T<:Mesh
    !((f.model==g.model)&&(f.mesh==g.mesh))&&throw(DomainError("SFMDistributions need the same model & mesh"))
    return SFMDistribution{T}(f.d+g.d,f.model,f.mesh)
end

size(d::SFMDistribution) = size(d.coeffs)
getindex(d::SFMDistribution,i::Int,j::Int) = d.coeffs[i,j]
setindex!(d::SFMDistribution,x,i::Int,j::Int) = throw(DomainError("inserted value(s) must be Float64"))
setindex!(d::SFMDistribution,x::Float64,i::Int,j::Int) = (d.coeffs[i,j]=x)
*(u::SFMDistribution,B::AbstractArray{Float64,2}) = SFMDistribution(*(u.coeffs,B),u.model,u.mesh)
*(B::AbstractArray{Float64,2},u::SFMDistribution) = *(u,B)
*(u::SFMDistribution,B::Number) = SFMDistribution(*(u.coeffs,B),u.model,u.mesh)
*(B::Number,u::SFMDistribution) = *(B,u)

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

function legendre_to_lagrange(coeffs)
    order = length(coeffs)
    V = vandermonde(n_bases(mesh))
    return V.V*coeffs
end

function pdf(d::SFMDistribution{T},model::Model) where T<:Mesh
    throw(DomainError("unknown SFMDistribution{<:Mesh}"))
end

function pdf(d::SFMDistribution{DGMesh},model::Model)
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            #fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,model,mesh)
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
pdf(d::SFMDistribution{T},model::Model,x,i) where T<:Mesh = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
pdf(d::SFMDistribution{T},model::Model,x::Float64,i::Int) where T<:Mesh = pdf(d,model)(x,i)
pdf(d::SFMDistribution{T},model::Model,x::Int,i::Int) where T<:Mesh = pdf(d,model)(convert(Float64,x),i)
# pdf(d::SFMDistribution{T},model::Model,x::Array{Float64,1},i::Int) where T<:Mesh = pdf(d,model).(x,i)
# pdf(d::SFMDistribution{T},model::Model,x::Array{Float64,1},i::Array{Int,1}) where T<:Mesh = pdf(d,model).(x,i)

function pdf(d::SFMDistribution{FRAPMesh},model::Model)
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,model,mesh)
            coeffs = d.coeffs[coeff_idx]
            # if not a point mass, then reconstruct solution
            # if !(cell_idx=="point mass")
                if _has_right_boundary(model.S,i)
                    yₖ₊₁ = mesh.nodes[cell_idx+1]
                    to_go = yₖ₊₁-x
                elseif _has_left_boundary(model.S,i)
                    yₖ = mesh.nodes[cell_idx]
                    to_go = x-yₖ
                end
                me = MakeME(CMEParams[n_bases(mesh)], mean = Δ(mesh)[cell_idx])
                fxi = (pdf(Array(coeffs'),me,to_go) + pdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx]-to_go))./cdf(Array(coeffs'),me,2*Δ(mesh)[cell_idx])
            # else 
            #     fxi = coeffs
            # end
        end
        return fxi
    end
    return f
end

function pdf(d::SFMDistribution{FVMesh},model::Model)
    function f(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        fxi = 0.0
        if ((x<=mesh.nodes[1])||(x>=mesh.nodes[end]))
            # fxi = 0.0
        else
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(x,i,model,mesh)
            # if not a point mass, then reconstruct solution
            # if !(cell_idx=="point mass")
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
                coeff_idx = (N₋(model.S) + (i-1)*total_n_bases(mesh)) .+ nodesIdx
                coeffs = d.coeffs[coeff_idx]#./Δ(mesh)[cell_idx]
                fxi = LinearAlgebra.dot(poly_vals,coeffs)
            # else 
            #     coeffs = d.coeffs[coeff_idx]
            #     fxi = coeffs
            # end
        end
        return fxi
    end
    return f
end

############
### CDFs ###
############

function cdf(d::SFMDistribution{T},model::Model) where T<:Mesh
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
            idx = (N₋(model.S) + (i-1)*total_n_bases(mesh) + (cell-1)*n_bases(mesh)) .+ 1 
            c += d.coeffs[idx]
        end
    else
        for cell in 1:(cell_idx-1)
            idx = (N₋(model.S) + (i-1)*total_n_bases(mesh) + (cell-1)*n_bases(mesh)) .+ (1:n_bases(mesh))
            c += sum(d.coeffs[idx])
        end
    end
    return c
end

function cdf(d::SFMDistribution{DGMesh},model::Model)
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, ~, ~ = _get_coeffs_index(xd,i,model,mesh)
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx, mesh, model)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d,model)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,n_bases(mesh))
                Fxi += quad
            end
            # add the RH point mass if  required
            if (x>=mesh.nodes[end])&&_has_right_boundary(model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,model)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end
cdf(d::SFMDistribution{T},model::Model,x,i) where T<:Mesh = 
    throw(DomainError("x must be Float64/Int/Array{Float64/Int,1}, i must be Int/Array{Int,1}"))
cdf(d::SFMDistribution{T},model::Model,x::Float64,i::Int) where T<:Mesh = cdf(d,model)(x,i)
cdf(d::SFMDistribution{T},model::Model,x::Int,i::Int) where T<:Mesh = cdf(d,model)(convert(Float64,x),i)

function cdf(d::SFMDistribution{FRAPMesh},model::Model)
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if _has_left_boundary(model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,model,mesh)
            coeffs = d.coeffs[coeff_idx]

            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left(d, i, cell_idx, mesh, model)
                
                # integrate up to x in the cell which contains x
                # me = MakeME(CMEParams[n_bases(mesh)], mean = Δ(mesh)[cell_idx])
                me = mesh.me
                a = Array(coeffs')
                if _has_right_boundary(model.S,i)
                    yₖ₊₁ = mesh.nodes[cell_idx+1]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (yₖ₊₁-xd)/Δ(mesh,cell_idx)
                        Fxi += mass*(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                elseif _has_left_boundary(model.S,i)
                    yₖ = mesh.nodes[cell_idx]
                    mass = sum(a)
                    if mass > 0
                        a = a./mass
                        to_go = (xd-yₖ)/Δ(mesh,cell_idx)
                        Fxi += mass*(1-(ccdf(a,me,to_go)-ccdf(a,me,2.0-to_go))/cdf(a,me,2.0))#sum(a) - (ccdf(a,me,to_go) - ccdf(a,me,2.0-to_go))/cdf(a,me,2.0)
                    end
                end
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,model)
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
        idx = N₋(model.S) + (i-1)*total_n_bases(mesh) + cell
        c += d.coeffs[idx]*Δ(mesh)[cell]
    end
    return c
end

function cdf(d::SFMDistribution{FVMesh},model::Model)
    function F(x::Float64,i::Int) # the PDF
        # check phase is in support 
        !(i∈phases(model)) && throw(DomainError("phase i must be in the support of the model"))
        # if x is not in the support return 0.0
        mesh = d.mesh
        Fxi = 0.0
        if (x<mesh.nodes[1])
            # Fxi = 0.0
        else
            # Fxi = 0.0
            # left pm
            if (x>=mesh.nodes[1])&&_has_left_boundary(model.S,i)
                ~, left_pm_idx = _get_point_mass_data_neg(i,mesh,model)
                left_pm = d.coeffs[left_pm_idx]
                Fxi += left_pm
            end
            # integral over density
            (x.>=mesh.nodes[end]) ? (xd=mesh.nodes[end]-sqrt(eps())) : xd = x
            cell_idx, cellnodes, coeff_idx = _get_coeffs_index(xd,i,model,mesh)
            # if not a point mass, then reconstruct solution
            if !(cell_idx=="point mass")
                # add all mass from cells to the left
                Fxi += _sum_cells_left_fv(d, i, cell_idx, mesh, model)

                # integrate up to x in the cell which contains x
                temp_pdf(y) = pdf(d,model)(y,i)
                quad = gauss_lobatto_quadrature(temp_pdf,mesh.nodes[cell_idx],xd,_order(mesh))
                Fxi += quad 
            end
            if (x>=mesh.nodes[end])&&_has_right_boundary(model.S,i)
                ~, right_pm_idx = _get_point_mass_data_pos(i,mesh,model)
                right_pm = d.coeffs[right_pm_idx]
                Fxi += right_pm
            end
        end
        return Fxi
    end
    return F
end


abstract type SFFMDistribution end

# """

#     SFFMDistribution(
#         pm::Array{<:Real},
#         distribution::Array{<:Real,3},
#         x::Array{<:Real},
#         type::String,
#     )

# - `pm::Array{Float64}`: a vector containing the point masses, the first
#     `sum(model.C.<=0)` entries are the left hand point masses and the last
#     `sum(model.C.>=0)` are the right-hand point masses.
# - `distribution::Array{Float64,3}`: "probability" or "density"` 
# - `x::Array{Float64,2}`:
#     - if `type="probability"` is a `1×NIntervals×NPhases` array
#         containing the cell centers.
#     - if `type="density"` is a `n_bases×NIntervals×NPhases` array
#         containing the cell nodes at which the denisty is evaluated.
# - `type::String`: either `"probability"` or `"density"`. `"cumulative"` is
#     not possible.
# """
struct SFFMDensity <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end
struct SFFMProbability <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end
struct SFFMCDF <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end

# """
# Convert from a vector of coefficients for the DG system to a distribution.

#     Coeffs2Dist(
#         model::Model,
#         mesh::Mesh,
#         Coeffs;
#         type::String = "probability",
#     )

# # Arguments
# - `model`: a Model object
# - `mesh`: a Mesh object as output from MakeMesh
# - `Coeffs::Array`: a vector of coefficients from the DG method
# - `type::String`: an (optional) declaration of what type of distribution you
#     want to convert to. Options are `"probability"` to return the probabilities
#     ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
#     return the CDF evaluated at cell edges, or `"density"` to return an
#     approximation to the density ar at the cell_nodes(mesh).

# # Output
# - a tuple with keys
# (pm=pm, distribution=yvals, x=xvals, type=type)
#     - `pm::Array{Float64}`: a vector containing the point masses, the first
#         `sum(model.C.<=0)` entries are the left hand point masses and the last
#         `sum(model.C.>=0)` are the right-hand point masses.
#     - `distribution::Array{Float64,3}`:
#         - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
#             containing the CDF evaluated at the cell edges as contained in
#             `x` below. i.e. `distribution[1,:,i]` returns the cdf at the
#             left-hand edges of the cells in phase `i` and `distribution[2,:,i]`
#             at the right hand edges.
#         - if `type="probability"` returns a `1×NIntervals×NPhases` array
#             containing the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
#             is the kth cell.
#         - if `type="density"` returns a `n_bases×NIntervals×NPhases` array
#             containing the density function evaluated at the cell nodes as
#             contained in `x` below.
#     - `x::Array{Float64,2}`:
#         - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
#             containing the cell edges as contained. i.e. `x[1,:]`
#             returns the left-hand edges of the cells and `x[2,:]` at the
#             right-hand edges.
#         - if `type="probability"` returns a `1×NIntervals×NPhases` array
#             containing the cell centers.
#         - if `type="density"` returns a `n_bases×NIntervals×NPhases` array
#             containing the cell nodes.
#     - `type`: as input in arguments.
# """
# function Coeffs2Dist(
#     model::Model,
#     mesh::DGMesh,
#     Coeffs::AbstractArray,
#     type::Type{T} = SFFMProbability,
#     v::Bool = false,
# ) where {T<:SFFMDistribution} 

#     V = vandermonde(n_bases(mesh))
#     N₋ = sum(model.C .<= 0)
#     N₊ = sum(model.C .>= 0)

#     if type == SFFMDensity
#         xvals = cell_nodes(mesh)
#         if basis(mesh) == "legendre"
#             yvals = reshape(Coeffs[N₋+1:end-N₊], n_bases(mesh), NIntervals(mesh), NPhases(model))
#             for i in 1:NPhases(model)
#                 yvals[:,:,i] = V.V * yvals[:,:,i]
#             end
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         elseif basis(mesh) == "lagrange"
#             yvals =
#                 Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, NIntervals(mesh) * NPhases(model)) .*
#                 (repeat(2.0 ./ Δ(mesh), 1, n_bases(mesh) * NPhases(model))'[:])
#             yvals = reshape(yvals, n_bases(mesh), NIntervals(mesh), NPhases(model))
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         end
#         if n_bases(mesh) == 1
#             yvals = [1;1].*yvals
#             xvals = [cell_nodes(mesh)-Δ(mesh)'/2;cell_nodes(mesh)+Δ(mesh)'/2]
#         end
#     elseif type == SFFMProbability
#         if n_bases(mesh) > 1 
#             xvals = cell_nodes(mesh)[1, :] + (Δ(mesh) ./ 2)
#         else
#             xvals = cell_nodes(mesh)
#         end
#         if basis(mesh) == "legendre"
#             yvals = (reshape(Coeffs[N₋+1:n_bases(mesh):end-N₊], 1, NIntervals(mesh), NPhases(model)).*Δ(mesh)')./sqrt(2)
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         elseif basis(mesh) == "lagrange"
#             yvals = sum(
#                 reshape(Coeffs[N₋+1:end-N₊], n_bases(mesh), NIntervals(mesh), NPhases(model)),
#                 dims = 1,
#             )
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         end
#     elseif type == SFFMCDF
#         if n_bases(mesh) > 1 
#             xvals = cell_nodes(mesh)[[1;end], :]
#         else
#             xvals = [cell_nodes(mesh)-Δ(mesh)'/2;cell_nodes(mesh)+Δ(mesh)'/2]
#         end
#         if basis(mesh) == "legendre"
#             tempDist = (reshape(Coeffs[N₋+1:n_bases(mesh):end-N₊], 1, NIntervals(mesh), NPhases(model)).*Δ(mesh)')./sqrt(2)
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         elseif basis(mesh) == "lagrange"
#             tempDist = sum(
#                 reshape(Coeffs[N₋+1:end-N₊], n_bases(mesh), NIntervals(mesh), NPhases(model)),
#                 dims = 1,
#             )
#             pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
#         end
#         tempDist = cumsum(tempDist,dims=2)
#         temppm = zeros(Float64,1,2,NPhases(model))
#         temppm[:,1,model.C.<=0] = pm[1:N₋]
#         temppm[:,2,model.C.>=0] = pm[N₊+1:end]
#         yvals = zeros(Float64,2,NIntervals(mesh),NPhases(model))
#         yvals[1,2:end,:] = tempDist[1,1:end-1,:]
#         yvals[2,:,:] = tempDist
#         yvals = yvals .+ reshape(temppm[1,1,:],1,1,NPhases(model))
#         pm[N₋+1:end] = pm[N₋+1:end] + yvals[end,end,model.C.>=0]
#     end
    
#     out = type(pm, yvals, xvals)
#     v && println("UPDATE: distribution object created with keys ", fieldnames(type))
#     return out
# end
# function Coeffs2Dist(
#     model::Model,
#     mesh::Union{FRAPMesh, FVMesh},
#     Coeffs::AbstractArray,
#     type::Type{T} = SFFMProbability,
#     v::Bool = false,
# ) where {T<:SFFMDistribution}

#     if type != SFFMProbability
#         args = [
#             model;
#             mesh;
#             Coeffs;
#             type;
#         ]
#         error("Input Error: no functionality other than 'probability' implemented, yet...")
#     end
    
#     N₋ = sum(model.C .<= 0)
#     N₊ = sum(model.C .>= 0)
    
#     xvals = cell_nodes(mesh)
    
#     yvals = sum(
#         reshape(Coeffs[N₋+1:end-N₊], n_bases(mesh), NIntervals(mesh), NPhases(model)),
#         dims = 1,
#     )
#     pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]

#     out = type(pm, yvals, xvals)
#     v && println("UPDATE: distribution object created with keys ", fieldnames(type))
#     return out
# end

# """
# Converts a distribution as output from `Coeffs2Dist()` to a vector of DG
# coefficients.

#     Dist2Coeffs(
#         model::Model,
#         mesh::Mesh,
#         Distn::SFFMDistribution,
#     )

# # Arguments
# - `model`: a Model object
# - `mesh`: a Mesh object as output from MakeMesh
# - `Distn::SFFMDistribution
#     - if `type="probability"` is a `1×NIntervals×NPhases` array containing
#         the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
#         is the kth cell.
#     - if `type="density"` is a `n_bases×NIntervals×NPhases` array containing
#         either the density function evaluated at the cell nodes which are in
#         `x` below, or, the inner product of the density function against the
#         lagrange polynomials.

# # Output
# - `coeffs` a row vector of coefficient values of length
#     `total_n_bases*NPhases + N₋ + N₊` ordered according to LH point masses, RH
#     point masses, interior basis functions according to basis function, cell,
#     phase. Used to premultiply operators such as B from `MakeB()`
# """
# function Dist2Coeffs(
#     model::Model,
#     mesh::DGMesh,
#     Distn::SFFMDensity,
# )
#     V = vandermonde(n_bases(mesh))
#     theDistribution =
#         zeros(Float64, n_bases(mesh), NIntervals(mesh), NPhases(model))
#     if basis(mesh) == "legendre"
#         theDistribution = Distn.distribution
#         for i = 1:NPhases(model)
#             theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
#         end
#     elseif basis(mesh) == "lagrange"
#         theDistribution .= Distn.distribution
#         # convert to probability coefficients by multiplying by the
#         # weights in V.w/2 and cell widths Δ
#         theDistribution = ((V.w .* theDistribution).*(Δ(mesh) / 2)')[:]
#     end
#     # also put the point masses on the ends
#     coeffs = [
#         Distn.pm[1:sum(model.C .<= 0)]
#         theDistribution[:]
#         Distn.pm[sum(model.C .<= 0)+1:end]
#     ]
#     coeffs = Matrix(coeffs[:]')
#     return coeffs
# end

# function Dist2Coeffs(
#     model::Model,
#     mesh::DGMesh,
#     Distn::SFFMProbability,
# )
#     V = vandermonde(n_bases(mesh))
#     theDistribution =
#         zeros(Float64, n_bases(mesh), NIntervals(mesh), NPhases(model))
#     if basis(mesh) == "legendre"
#         # for the legendre basis the first basis function is ϕ(x)=Δ√2 and
#         # all other basis functions are orthogonal to this. Hence, we map
#         # the cell probabilities to the first basis function only.
#         theDistribution[1, :, :] = Distn.distribution./Δ(mesh)'.*sqrt(2)
#     elseif basis(mesh) == "lagrange"
#         theDistribution .= Distn.distribution
#         # convert to probability coefficients by multiplying by the
#         # weights in V.w/2
#         theDistribution = (V.w .* theDistribution / 2)[:]
#     end
#     # also put the point masses on the ends
#     coeffs = [
#         Distn.pm[1:sum(model.C .<= 0)]
#         theDistribution[:]
#         Distn.pm[sum(model.C .<= 0)+1:end]
#     ]
#     coeffs = Matrix(coeffs[:]')
#     return coeffs
# end
# function Dist2Coeffs(
#     model::Model,
#     mesh::Union{FRAPMesh,FVMesh},
#     Distn::SFFMDistribution
# )
    
#     # also put the point masses on the ends
#     coeffs = [
#         Distn.pm[1:sum(model.C .<= 0)]
#         Distn.distribution[:]
#         Distn.pm[sum(model.C .<= 0)+1:end]
#     ]
    
#     coeffs = Matrix(coeffs[:]')
#     return coeffs
# end

# """
# Computes the error between distributions.

#     starSeminorm(d1::SFFMProbability, d2::SFFMProbability)

# # Arguments
# - `d1`: a distribution object as output from `Coeffs2Dist` 
# - `d2`: a distribution object as output from `Coeffs2Dist` 
# """
# function starSeminorm(d1::SFFMProbability, d2::SFFMProbability)
#     e = sum(abs.(d1.pm-d2.pm)) + sum(abs.(d1.distribution-d2.distribution))
#     return e
# end
