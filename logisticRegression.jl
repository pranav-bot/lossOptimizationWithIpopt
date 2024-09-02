include("lossFunctions.jl")
include("encoders.jl")
include("activationFunctions.jl")
using CSV, GLM, Plots, TypedTables, Statistics, Ipopt, JuMP

################################################################################
# Models
################################################################################

function logisticRegression(x, y, epochs, α)
    b = 0.0  # intercept or bias
    m = 0.0  # slope
    loss_history = Float64[]
    best_loss = Inf
    b_history = Float64[]  # Record history of b
    m_history = Float64[]  # Record history of m

    for epoch in 1:epochs
        ŷ = b .+ m .* x
        ŷ = sigmoid(ŷ)
        loss = binaryCrossEntropy(y, ŷ)
        push!(loss_history, loss)
        push!(b_history, b)  # Record current b
        push!(m_history, m)  # Record current m

        # Calculate gradients
        residuals = ŷ - y
        gradient_b = sum(residuals) / length(x)
        gradient_m = (1 / length(x)) * sum(residuals .* x)

        # Update parameters
        b -= α * gradient_b
        m -= α * gradient_m

        if loss < best_loss
            best_loss = loss
        end
    end
    ŷ = b .+ m .* x  # equation of the line
    ŷ = sigmoid(ŷ)
    final_loss = binaryCrossEntropy(y, ŷ)
    push!(loss_history, final_loss)

    println("Optimized Slope Batch Gradient Descent (m): ", m)
    println("Optimized Intercept Batch Gradient Descent (b): ", b)
    println("Final Loss Batch Gradient Descent: ", best_loss)
    println("\n")
    
    return m, b, best_loss, loss_history, b_history, m_history
end

function logisticRegressionWithIpopt(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    model = Model(Ipopt.Optimizer)

    # Register the length function with one argument to avoid the warning
    register(model, :length, 1, length; autodiff = true)

    # Define variables for intercept (b) and slope (m)
    @variable(model, b, start = 0.0)
    @variable(model, m, start = 0.0)

    # Define the hypothesis for each data point as a scalar expression
    @NLexpression(model, ŷ[i=1:length(x)], 1 / (1 + exp(- (b + m * x[i]))))

    # Define the objective function (binary cross-entropy) as a scalar sum
    @NLobjective(model, Min, (-1/n)*sum(y[i] * log(ŷ[i] + 1e-10) + (1 - y[i]) * log(1 - ŷ[i] + 1e-10) for i in 1:length(x)))

    # Solve the optimization problem
    optimize!(model)

    # Extract optimized values of b and m
    b_opt = value(b)
    m_opt = value(m)

    # Calculate the final loss using scalar operations
    final_loss = - (1/length(x)) * sum(y[i] * log(1 / (1 + exp(- (b_opt + m_opt * x[i]))) + 1e-10) + 
                                        (1 - y[i]) * log(1 / (1 + exp(- (b_opt + m_opt * x[i]))) + 1e-10) 
                                        for i in 1:length(x))

    println("Optimized Intercept (b) Ipopt: ", b_opt)
    println("Optimized Slope (m) Ipopt: ", m_opt)
    println("Final Loss Ipopt: ", final_loss)

    return m_opt, b_opt, final_loss
end


################################################################################
# Test 1
################################################################################
println("Test 1:\n")

data = CSV.File("data/wolfspider.csv")

x = data.feature

y = data.class

y = binaryEncoder(y, "present")

epochs = 1000
learning_rate = 0.5
# Gradient Descent
m_gd, b_gd, best_loss_gd, loss_history_gd, b_history_gd, m_history_gd = logisticRegression(x, y, epochs, learning_rate)

# Ipopt
m_ipopt, b_ipopt, best_loss_ipopt = logisticRegressionWithIpopt(x, y)

# Decision Boundaries
x_range = minimum(x):0.01:maximum(x)
y_pred_gd = sigmoid.(b_gd .+ m_gd .* x_range)
y_pred_ipopt = sigmoid.(b_ipopt .+ m_ipopt .* x_range)

scatter(x, y, label="Data", xlabel="Feature", ylabel="Class", title="Logistic Regression Decision Boundary")
plot!(x_range, y_pred_gd, label="GD Decision Boundary", color=:red, linewidth=2)
plot!(x_range, y_pred_ipopt, label="Ipopt Decision Boundary", color=:green, linewidth=2)
savefig("./plots/DecisionBoundaryComparison.png")

# Learning Curves
p_learning_curve = plot(1:epochs, loss_history_gd[1:end-1],  # Exclude the last entry since it was added after the loop
    xlabel="Epochs",
    ylabel="Loss",
    title="Learning Curve",
    label="GD Loss",
    color=:blue,
    linewidth=2
)
savefig("./plots/LearningCurveGD.png")

# Gradient Descent Path
p_gradient_path = scatter(m_history_gd, b_history_gd,
    xlabel="Slope (m)",
    ylabel="Intercept (b)",
    title="Gradient Descent Path",
    color=:blue,
    alpha=0.5,
    label="GD Path"
)
savefig("./plots/GradientDescentPathGD.png")