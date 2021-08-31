function lagrange_poly_basis(nodes, evalPt)
    order = length(nodes)
    poly_coefs = zeros(order)
    for n in 1:order
        notn = [1:n-1;n+1:order]
        poly_coefs[n] = prod(evalPt.-nodes[notn])./prod(nodes[n].-nodes[notn])
    end
    return poly_coefs
end

function lagrange_interpolant()

end
function gauss_lobatto_quadrature(fun::Function,a::Float64,b::Float64,n_evals::Int)
    (b<=a)&&throw(DomainError("must have a<b"))
    (n_evals<1)&&throw(DomainError("n_evals must be > 0"))
    if n_evals > 1
        # the GL nodes
        # nodes in [-1,1]
        nodes = Jacobi.zglj(n_evals, 0, 0)
        weights = Jacobi.wglj(nodes,0,0)*(b-a)/2
        # shift nodes to [a,b]
        nodes *= 0.5*(b-a)
        nodes .+= 0.5*(a+b)
    else
        nodes = 0.5*(a+b)
        weights = (b-a)
    end
    fun_vals = fun.(nodes)
    quad = LinearAlgebra.dot(fun_vals,weights)
end