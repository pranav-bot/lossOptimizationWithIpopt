include("lossFunctions.jl")
using CSV, GLM, Plots, TypedTables, Statistics, Ipopt, JuMP

################################################################################
# Models
################################################################################

function logisticRegression(x, y)
    b = 0
    m = 0
    ŷ = b .+ m.*x
end