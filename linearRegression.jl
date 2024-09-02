include("lossFunctions.jl")
include("plots.jl")
using CSV, GLM, Plots, TypedTables, Statistics, Ipopt, JuMP

################################################################################
# Models
################################################################################
function linearRegressionWithOLS(x, y)
    t = Table(X=x, Y=y)
    ols = lm(@formula(Y~X), t)
    return ols
end 

function linearRegressionWithBatchGradientDescentWeightsZeroInitialized(x, y, epochs, α)
    b = 0.0  # intercept or bias
    m = 0.0  # slope
    loss_history = Float64[]
    best_loss = Inf
    b_history = Float64[]  # Record history of b
    m_history = Float64[]  # Record history of m
    
    for epoch in 1:epochs
        ŷ = b .+ m .* x  # equation of the line
        loss = MeanSquaredError(y, ŷ)
        push!(loss_history, loss)
        push!(b_history, b)  # Record current b
        push!(m_history, m)  # Record current m
        
        # Calculate gradients
        residuals = ŷ - y
        gradient_b = sum(residuals)/length(x)
        gradient_m = (1 / length(x)) * sum(residuals .* x)
        
        # debugging
        #println("Epoch: ", epoch)
        #println("Gradient b: ", gradient_b)
        #println("Gradient m: ", gradient_m)
        
        # Update parameters
        b -= α * gradient_b
        m -= α * gradient_m
        
        # debugging
        #println("Updated b: ", b)
        #println("Updated m: ", m)
        
        # Finding best loss
        if loss < best_loss
            best_loss = loss
        end
    end
    
    # Final predictions and loss
    ŷ = b .+ m .* x  # equation of the line
    final_loss = MeanSquaredError(y, ŷ)
    push!(loss_history, final_loss)

    println("Optimized Slope Batch Gradient Descent (m): ", m)
    println("Optimized Intercept Batch Gradient Descent (b): ", b)
    println("Final Loss Batch Gradient Descent: ", best_loss)
    println("\n")
    
    return m, b, best_loss, loss_history, b_history, m_history
end

function linearRegressionWithIpopt(x, y)
    # JuMP model initialization with IPOPT as the optimizer
    model = Model(Ipopt.Optimizer)

    # Define variables for the bias (b) and slope (m) with initial values
    @variable(model, b, start = 0.0)  
    @variable(model, m, start = 0.0) 

    # Define the objective function to minimize the sum of squared errors
    @objective(model, Min, MeanSquaredError(y, [b + m * xi for xi in x]))

    # Solve the optimization problem
    optimize!(model)

    # Extract optimized values of b and m
    b_opt = value(b)
    m_opt = value(m)

    # Calculate the final predictions and loss
    ŷ = b_opt .+ m_opt .* x
    final_loss = MeanSquaredError(y, ŷ)

    # Print debugging information
    println("Optimized b Ipopt: ", b_opt)
    println("Optimized m Ipopt: ", m_opt)
    println("Final Loss Ipopt: ", final_loss)

    return m_opt, b_opt, final_loss
end

################################################################################
# Test 1
################################################################################
println("Test 1:\n")
data = CSV.File("data/housingdata.csv")

ols = linearRegressionWithOLS(data.size, round.(Int, data.price/1000))
predictions_ols = predict(ols)
m, b, loss, loss_history, b_history, m_history = linearRegressionWithBatchGradientDescentWeightsZeroInitialized(
    data.size,
    round.(Int, data.price/1000), 
    100,
    0.0000001   
)

m_ipopt, b_ipopt, loss_ipopt = linearRegressionWithIpopt(
    data.size,
    round.(Int, data.price / 1000)
)

plot(data.size, round.(Int, data.price / 1000), seriestype = :scatter, label = "Data", title = "Linear Regression with OLS Housing Portland Data")
plot!(data.size, predictions_ols, label = "Fitted Line (OLS)", color = :green, linewidth = 2)
plot!(data.size, b .+ m .* data.size, label = "Fitted Line(BGD)", color = :red, linewidth = 2)
plot!(data.size, b_ipopt .+ m_ipopt .* data.size, label = "Fitted Line (IPOPT)", color = :blue, linewidth = 2)
savefig("./plots/Test1.png")
plot_loss_surface(data.size, round.(Int, data.price / 1000), b_history, m_history)

################################################################################
# Test 2
################################################################################
println("Test 2:\n")
x = 1:20
y = 4*x + 8*(0.5.-rand(20))
ols = linearRegressionWithOLS(x, y)
predictions_ols = predict(ols)

m, b, loss, loss_history = linearRegressionWithBatchGradientDescentWeightsZeroInitialized(
    x,
    y, 
    100,
    0.01
)

m_ipopt, b_ipopt, loss_ipopt = linearRegressionWithIpopt(
    x,
    y
)

plot(x, y, seriestype = :scatter, label = "Data", title = "Linear Regression with OLS Regression Data")
plot!(x, predictions_ols, label = "Fitted Line (OLS)", color = :green, linewidth = 2)
plot!(x, b .+ m .* x, label = "Fitted Line (BGD)", color = :red, linewidth = 2)
plot!(x, b_ipopt .+ m_ipopt * x, label = "Fitted Line (IPOPT)", color = :blue, linewidth = 2)
savefig("./plots/Test2.png")

################################################################################
# Benchmarking
################################################################################