function binaryEncoder(y, category1)
    for i in 1:eachindex(y)
        if y[i] == category1
            y[i] = 1.0
        else
            y[i] = 0.0
        end    
    end
end