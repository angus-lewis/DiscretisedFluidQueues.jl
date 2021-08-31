using JSON, LinearAlgebra, JLD2

# read the data from json
tempCMEParams = Dict()
open("src/CMEParamsData/iltcme.json", "r") do f #iltcme.json is from Horvath, Telek, see CMEParams_metadata, or fullData.zip if you have it
    global tempCMEParams
    tempCMEParams=JSON.parse(f)  # parse and transform data
end

# put the parameters in CMEParams with keys corresponding the ME order
CMEParams = Dict()
for n in keys(tempCMEParams)
    if !in(2*tempCMEParams[n]["n"]+1,keys(CMEParams)) # if no already in dict add it
        CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
    elseif tempCMEParams[n]["cv2"]<CMEParams[2*tempCMEParams[n]["n"]+1]["cv2"]
        # if its already in there, add only if it has the smallest CV
        CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
    end
end

# a function to numerically approximate D
function integrateD(evals,params)
    # evals is an integer specifying how many points to eval the function at
    # params is a CMEParams dictionary entry, i.e. CMEParams[3]
    N = 2*params["n"]+1 # ME order

    α = zeros(N)
    α[1] = params["c"]
    a = params["a"]
    b = params["b"]
    ω =  params["omega"]
    for k in 1:params["n"]
        kω = k*ω
        α[2*k] = (1/2)*( a[k]*(1+kω) - b[k]*(1-kω) )/(1+kω^2)
        α[2*k+1] = (1/2)*( a[k]*(1-kω) + b[k]*(1+kω) )/(1+kω^2)
    end

    period = 2*π/ω # the orbit repeats after this time
    edges = range(0,period,length=evals+1) # points at which to evaluate the fn
    h = period/(evals)

    orbit_LHS = α
    orbit_RHS = zeros(N)
    v_RHS = zeros(N)
    v_RHS[1] = 1
    v_LHS = ones(N)
    D = zeros(N,N)
    for t in edges[2:end]
        orbit_RHS[1] = α[1]
        for k in 1:params["n"]
            kωt = k*ω*t
            idx = 2*k
            idx2 = idx+1
            temp_cos = cos(kωt)
            temp_sin = sin(kωt)
            orbit_RHS[idx] = α[idx]*temp_cos + α[idx2]*temp_sin
            orbit_RHS[idx2] = -α[idx]*temp_sin + α[idx2]*temp_cos
            v_RHS[idx] = temp_cos - temp_sin
            v_RHS[idx2] = temp_sin + temp_cos
        end
        orbit_RHS = orbit_RHS./sum(orbit_RHS)
        orbit = (orbit_LHS+orbit_RHS)./2

        v = exp(-(t-h))*(v_LHS - exp(-h)*v_RHS)

        Dᵢ = v*orbit'
        D += Dᵢ

        orbit_LHS = copy(orbit_RHS)
        v_LHS = copy(v_RHS)
    end
    D = (1/(1-exp(-period)))*D
    return D
end

# numerical approximation of D
k = 100_000_000
T=[]
N=[]
for n in keys(CMEParams)
    evals = Int(ceil(k/(n/3)))# number of function evals
    display(n)
    display(evals)
    CMEParams[n]["D"], t = @timed integrateD(evals,CMEParams[n])
    CMEParams[n]["intDevals"] = evals
    display(t)
    push!(N,n)
    push!(T,t)
end

scatter(N,T)

# add order 1 (exponential) to params
CMEParams[1] = Dict(
  "n"       => 0,
  "c"       => 1,
  "b"       => Any[],
  "mu2"     => 1,
  "a"       => Any[],
  "omega"   => 1,
  "phi"     => [],
  "mu1"     => 1,
  "D"       => [1.0],
  "cv2"     => 1,
  "optim"   => "full",
  "lognorm" => [],
)

open("dev/CMEParams.json","w") do f
    JSON.print(f, CMEParams)
end

@save "dev/CMEParams.jld2" CMEParams

let
    CMEKeys = sort(collect(keys(CMEParams)))
    a = 0
    filecounter = 1
    tempDict = Dict()
    for key in CMEKeys
        a += key^2*8
        tempDict[key] = CMEParams[key]
        if a > 2e7
            open(pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".json","w") do f
                JSON.print(f, tempDict)
            end
            @save pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".jld2" tempDict
            tempDict = Dict()
            a = 0
            filecounter += 1
        end
    end
end
