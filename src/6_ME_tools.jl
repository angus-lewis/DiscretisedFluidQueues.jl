import FileIO, LinearAlgebra, JSON

# erlangDParams = Dict()
src_dir = pwd()*"/StochasticFluidQueues/src/"
FileIO.load(src_dir*"erlangParamsData/erlangDParams.jld2", "erlangDParams")

FileIO.load(src_dir*"/CMEParamsData/CMEParams.jld2", "CMEParams")

"""
ME constructor method
    
    ME(
        a::Array{<:Real,2},
        S::Array{<:Real,2},
        s::Array{<:Real,1},
        D::Array{<:Real}=[0],
    )

Inputs: 
 - `a` a 1 by p Array of reals
 - `S` a p by p Array of reals
 - `s` a p by 1 Array of reals
 - `D` an optional argument, if empty then me.D is the identity, 
    else is a p by p matrix
 Throws an error if the dimensions are inconsistent.
"""
struct ME 
    a::Array{<:Real,2}
    S::Array{<:Real,2}
    s::Union{Array{<:Real,1},Array{<:Real,2}}
    D::Array{<:Real,2}
    function ME(
        a::Array{<:Real,2},
        S::Array{<:Real,2},
        s::Union{Array{<:Real,1},Array{<:Real,2}};
        D::Array{<:Real,2}=zeros(1,1),
    )
    
        if D==zeros(1,1)
            D = Array{Float64}(LinearAlgebra.I(size(S,1)))
        end
        s1 = size(a,1)
        s2 = size(a,2)
        s3 = size(S,1)
        s4 = size(S,2)
        s5 = size(s,1)
        s6 = size(s,2)
        s7 = size(D,1)
        s8 = size(D,2)
        checksquare(D,"D")
        checksquare(S,"S")
        test = (s1!=1) || (s6!=1) || any(([s2;s3;s4;s7;s8].-s5).!=0)
        if test
            error("Dimensions of ME representation not consistent")
        else
            return new(a,S,s,D)
        end
    end
end

pdf(me::ME) = x->(me.a*exp(me.S*x)*me.s)[1]
pdf(a::Array{<:Real,2}, me::ME) = 
    (length(a)==size(me.S,1)) ? (x->(a*exp(me.S*x)*me.s)[1]) : throw(
        DomainError("a and me.S must have compatible size"))

pdf(me::ME, x::Real) = pdf(me)(x)
pdf(a::Array{<:Real,2}, me::ME, x::Real) = pdf(a,me)(x)

pdf(me::ME, x::Array{<:Real}) = pdf(me).(x)
pdf(a::Array{<:Real,2}, me::ME, x::Array{<:Real}) = pdf(a,me).(x)

ccdf(me::ME) = x->sum(me.a*exp(me.S*x))
ccdf(a::Array{<:Real,2}, me::ME) = (length(a)==size(me.S,1)) ? (x->sum(a*exp(me.S*x))) : throw(
    DomainError("a and me.S must have compatible size"))

ccdf(me::ME, x::Real) = ccdf(me)(x)
ccdf(a::Array{<:Real,2}, me::ME, x::Real) = ccdf(a,me)(x)

ccdf(me::ME, x::Array{<:Real}) = ccdf(me).(x)
ccdf(a::Array{<:Real,2}, me::ME, x::Array{<:Real}) = ccdf(a,me).(x)

cdf(me::ME) = x->1-ccdf(me,x)
cdf(a::Array{<:Real,2}, me::ME) = x->1-ccdf(a,me,x)

cdf(me::ME, x::Real) = cdf(me)(x)
cdf(a::Array{<:Real,2}, me::ME, x::Real) = cdf(a,me)(x)

cdf(me::ME, x::Array{<:Real}) = cdf(me).(x)
cdf(a::Array{<:Real,2}, me::ME, x::Array{<:Real}) = cdf(a,me).(x)


"""

"""
function MakeME(params; mean::Real = 1)
    N = 2*params["n"]+1
    α = zeros(1,N)
    α[1] = params["c"]
    a = params["a"]
    b = params["b"]
    ω =  params["omega"]
    for k in 1:params["n"]
        kω = k*ω
        α[2*k] = (1/2)*( a[k]*(1+kω) - b[k]*(1-kω) )/(1+kω^2)
        α[2*k+1] = (1/2)*( a[k]*(1-kω) + b[k]*(1+kω) )/(1+kω^2)
    end
    α = α./sum(α)
    Q = zeros(N,N)
    Q[1,1] = -1
    for k in 1:params["n"]
        kω = k*ω
        idx = 2*k:(2*k+1)
        Q[idx,idx] = [-1 -kω; kω -1]
    end
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)
    return ME(α,Q,q;D=params["D"])
end

function MakeErlang(order; mean::Real = 1)
    α = zeros(1,order) # inital distribution
    α[1] = 1
    λ = order/mean
    Q = zeros(order,order)
    Q = Q + LinearAlgebra.diagm(0=>repeat(-[λ],order), 1=>repeat([λ],order-1))
    q = -sum(Q,dims=2)
    D = Array{Float64}(LinearAlgebra.I(order))[end:-1:1,:]
    return ME(α,Q,q;D=D)
end

orbit(t,me::ME; norm = 1) = begin
    orbits = zeros(length(t),length(me.a))
    for i in 1:length(t)
        num = me.a*exp(me.S*t[i])
        if norm == 1
            denom = sum(num)
        elseif norm == 2
            denom = exp(me.S[1,1]*t[i])#sum(num)
        else
            denom = 1
        end
        orbits[i,:] = num./denom
    end
    return orbits
end

## not used, but kept them because they might be / are cool 
# function renewalProperties(me::ME)
#     density(t) = begin
#         Q = me.S
#         q = me.s
#         α = me.a
#         e = ones(size(q))
#         (α*exp((Q+q*α)*t)*q)[1]
#     end
#     mean(t) = begin
#         Q = me.S
#         q = me.s
#         α = me.a
#         e = ones(size(q))
#         temp1 = α*-Q^-1
#         temp2 = temp1*e
#         temp3 = temp1./temp2
#         temp4 = Q + (q*α)
#         ((t./temp2) - α*(I - exp(temp4*t))*(temp4 + e*temp3)^-1*q)[1]
#     end
#     ExpectedOrbit(t) = begin
#         Q = me.S
#         q = me.s
#         α = me.a
#         e = ones(size(q))
#         (α*exp((Q + q*α)*t))[1]
#     end
#     return (density=density,mean=mean,ExpectedOrbit=ExpectedOrbit)
# end
