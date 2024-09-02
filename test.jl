using CSV, GLM, Plots, TypedTables

data = CSV.File("data/housingdata.csv")

X = data.size

y = round.(Int, data.price/1000)

t = Table(X = X, Y = y)

gr(size = (600,600))

p_scatter = scatter(X, y, 
    xlims = (0,5000), 
    ylims = (0,800),
    xlabel = "Size(sqft)",
    ylabel = "Price",
    title = "Housing",
    legend = true,
    color = :red
    )
 
ols = lm(@formula(Y~X), t)


a = [1,2]

b = [2,2]

m=2

print(a.*b)