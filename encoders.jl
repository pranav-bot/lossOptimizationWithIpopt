function binaryEncoder(y, category1)
    y_temp = Float64[]
    for i in 1:length(y)
        if y[i] == category1
            a = 1.0
        else
            a = 0.0
        end
        push!(y_temp, a)    
    end
    return y_temp
end

function encodemake(x::Array)
    uniques = unique(x)
    dict = Dict(item => Float64(idx) for (idx,item) in enumerate(uniques))
    values_array = [dict[item] for item in x]
    return values_array
end