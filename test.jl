include("lossFunctions.jl")
using CSV, GLM, Plots, TypedTables, Statistics

function linearRegressionWithOLS(x, y)
    t = Table(X=x, Y=y)
    ols = lm(@formula(Y~X), t)
    return ols
end 

# Define Mean Squared Error function if not defined elsewhere
function MeanSquaredError(y_true, y_pred)
    return mean((y_pred .- y_true) .^ 2)
end

function linearRegressionWithBatchGradientDescentWeightsZeroInitialized(x, y; epochs=100, α=0.0000001)
    b = 0.0  # intercept or bias
    m = 0.0  # slope
    loss_history = Float64[]
    
    for epoch in 1:epochs
        ŷ = b .+ m .* x  # equation of the line
        loss = MeanSquaredError(y, ŷ)
        push!(loss_history, loss)
        
        # Calculate gradients
        residuals = ŷ - y
        gradient_b = mean(residuals)
        gradient_m = (1 / length(x)) * sum(residuals .* x)
        
        # Update parameters
        b -= α * gradient_b
        m -= α * gradient_m
    end
    
    # Final predictions and loss
    ŷ = b .+ m .* x  # equation of the line
    final_loss = MeanSquaredError(y, ŷ)
    push!(loss_history, final_loss)
    
    return m, b, final_loss, loss_history
end

# Example usage
x = 1:20
y = 4 .* x .+ 8 .* (0.5 .- rand(20))

# OLS Regression
ols = linearRegressionWithOLS(x, y)
predictions_ols = predict(ols)

# Batch Gradient Descent
m, b, loss, loss_history = linearRegressionWithBatchGradientDescentWeightsZeroInitialized(
    x,
    y, 
    epochs=100,
    α=0.0000001
)

println("Optimized Slope (m): ", m)
println("Optimized Intercept (b): ", b)
println("Final Loss: ", loss)

# Plotting
plot(x, y, seriestype = :scatter, label = "Data", title = "Linear Regression Comparison")
plot!(x, predictions_ols, label = "Fitted Line (OLS)", color = :green, linewidth = 2)
plot!(x, b .+ m .* x, label = "Fitted Line (BGD)", color = :red, linewidth = 2)
