using JSON, JLD2

CMEParams = Dict()
for i in 1:26
    @load pwd()*"/src/CMEParamsData/CMEParams"*string(i)*".jld2" tempDict
    for k in keys(tempDict)
        if k < 50
            CMEParams[k] = tempDict[k]
        end
    end
end

open("src/CMEParamsData/CMEParams.json","w") do f
    JSON.print(f, CMEParams)
end

@save "src/CMEParamsData/CMEParams.jld2" CMEParams
