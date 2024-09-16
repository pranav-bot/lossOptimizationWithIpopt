using CategoricalArrays
using DataFrames

# Create a DataFrame
df = DataFrame(CategoryColumn = ["A", "B", "A", "C", "B", "C"])

# Display original DataFrame
println("Original DataFrame:")
println(df)

# Convert to Categorical Data
df.CategoryColumn = CategoricalArray(df.CategoryColumn)

# Display encoded DataFrame
println("Encoded DataFrame:")
println(df)

# Display the type of the column
println("Column type:")
println(eltype(df.CategoryColumn))

# Retrieve numerical labels (codes)
println("Numerical labels:")
println(df.CategoryColumn.codes)

# Retrieve original categories
println("Original categories:")
println(levels(df.CategoryColumn))
