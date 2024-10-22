using DataFrames

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

function encodemake(x::Any)
    uniques = unique(x)
    dict = Dict(item => Float64(idx) for (idx,item) in enumerate(uniques))
    values_array = [dict[item] for item in x]
    return values_array
end

# function label_encode(df::DataFrame, column::Symbol)
#     # Ensure the column exists in the DataFrame
#     if !haskey(df, column)
#         throw(ArgumentError("Column $column does not exist in the DataFrame."))
#     end
    
#     # Extract the unique categories from the column
#     unique_categories = unique(df[!, column])
    
#     # Create a dictionary mapping categories to integers
#     category_to_int = Dict(cat => i for (i, cat) in enumerate(unique_categories))
    
#     # Apply the mapping to encode the column
#     df[!, column] = [category_to_int[cat] for cat in df[!, column]]
    
#     # Return the mapping for reference
#     return category_to_int
# end