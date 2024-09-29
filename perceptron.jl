include("activationFunctions.jl")
include("lossFunctions.jl")
include("encoders.jl")
using CSV, DataFrames, LinearAlgebra, Random, Plots, Flux, JuMP, Ipopt
using CategoricalArrays

mutable struct Perceptron
    lr::Float64
    weights::Vector{Float64}
    bias::Float64
    epochs::Int64
    w_history::Vector{Vector{Float64}}
    b_history::Vector{Float64}
    loss_history::Vector{Float64}
    best_w::Vector{Float64}
    best_b::Float64
    best_loss::Float64
end

function Perceptron(lr::Float64, n_features::Int64, epochs::Int64)
    weights = zeros(n_features)
    bias = 0.0
    w_history = Vector{Vector{Float64}}()
    b_history = Vector{Float64}()
    loss_history = Vector{Float64}()
    best_w = copy(weights)
    best_b = bias
    best_loss = Inf

    return Perceptron(lr, weights, bias, epochs, w_history, b_history, loss_history, best_w, best_b, best_loss)
end

function fit_zero_init_weights_SGD(perceptorn::Perceptron, X, y)
    perceptorn.weights = zeros(size(X)[2])
    perceptorn.bias = 0
    total_loss = 0

    for _ in 1:perceptorn.epochs
        for i in 1:size(X)[1]
            y_pred = sigmoid(perceptorn.bias + dot(perceptorn.weights, X[i,:]))
            perceptorn.weights = perceptorn.weights .+ perceptorn.lr *(y[i]-y_pred).*X[i,:]
            perceptorn.bias = perceptorn.bias .+ perceptorn.lr *(y[i]-y_pred)
            # Compute the loss after the entire epoch
            total_loss = Flux.binarycrossentropy(y_pred, y) 
        end

        push!(perceptorn.w_history, copy(perceptorn.weights))
        push!(perceptorn.b_history, perceptorn.bias)
        push!(perceptorn.loss_history, total_loss)
        if total_loss < perceptorn.best_loss
            perceptorn.best_loss = total_loss
            perceptorn.best_w .= perceptorn.weights
            perceptorn.best_b = perceptorn.bias
        end
    end
    return perceptorn
end

function fit_random_init_weights_SGD(perceptorn::Perceptron, X, y)
    perceptorn.weights = rand(Float64, size(X)[2]) 
    perceptorn.bias = 0
    total_loss = 0

    for _ in 1:perceptorn.epochs
        for i in 1:size(X)[1]
            #println("I", i)
            y_pred = sigmoid(perceptorn.bias + dot(perceptorn.weights, X[i,:]))
            perceptorn.weights = perceptorn.weights .+ perceptorn.lr *(y[i]-y_pred).*X[i,:]
            perceptorn.bias = perceptorn.bias .+ perceptorn.lr *(y[i]-y_pred) 
        end
        # Compute the loss after the entire epoch
        total_loss = Flux.binarycrossentropy(y_pred, y)
        push!(perceptorn.w_history, copy(perceptorn.weights))
        push!(perceptorn.b_history, perceptorn.bias)
        push!(perceptorn.loss_history, total_loss)
        if total_loss < perceptorn.best_loss
            perceptorn.best_loss = total_loss
            perceptorn.best_w .= perceptorn.weights
            perceptorn.best_b = perceptorn.bias
        end
    end
    return perceptorn
end

function optimize_perceptron_jump(X::Matrix{Float64}, y::Vector{Int64}, epochs::Int64=1000, λ::Float64=0.01)
    n_features = size(X, 2)
    n_samples = size(X, 1)

    # Feature scaling (standardization)
    #X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

    # Create a JuMP model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Define variables
    @variable(model, w[1:n_features])  # weights
    @variable(model, b)                # bias

    # Define auxiliary variables for the sigmoid function
    @variable(model, z[1:n_samples])
    @variable(model, p[1:n_samples])

    # Constraints for z (linear combination)
    @constraint(model, [i=1:n_samples], z[i] == sum(X[i,j] * w[j] for j in 1:n_features) + b)

    # Exact sigmoid function
    @NLconstraint(model, [i=1:n_samples], p[i] == 1 / (1 + exp(-z[i])))

    # Objective: Minimize binary cross-entropy loss with L2 regularization
    @NLobjective(model, Min, 
        sum(-y[i] * log(p[i] + 1e-10) - (1 - y[i]) * log(1 - p[i] + 1e-10) for i in 1:n_samples) / n_samples
        + λ * sum(w[j]^2 for j in 1:n_features)  # L2 regularization term
    )

    # Set the maximum number of iterations
    set_optimizer_attribute(model, "max_iter", epochs)

    # Solve the optimization problem
    optimize!(model)

    # Extract results
    optimal_weights = value.(w)
    optimal_bias = value(b)
    optimal_loss = objective_value(model)
    status = termination_status(model)

    return optimal_weights, optimal_bias, optimal_loss, status
end


function predict(perceptron::Perceptron, X)
    y_pred = zeros(size(X, 1))
    for i in 1:size(X, 1)
        prob = sigmoid(perceptron.bias + dot(perceptron.weights, X[i, :]))
        y_pred[i] = prob >= 0.5 ? 1 : 0  # Threshold at 0.5 for binary classification
    end
    return y_pred
end

function accuracy(y_true, y_pred)
    return sum(y_true .== y_pred) / length(y_true)
end

df = CSV.read("data/Titanic-Dataset.csv", DataFrame)
df = select!(df, Not(:PassengerId, :Name, :Ticket, :Cabin))
df.Sex = encodemake(df.Sex)
df.Embarked = encodemake(df.Embarked)
df = dropmissing!(df)
X = Matrix(df)
y = df.Survived


perceptron = Perceptron(0.01, size(X)[2], 1000)

trained_perceptron  = fit_zero_init_weights_SGD(perceptron, X, y)

y_pred = predict(trained_perceptron, X)

#Calculate accuracy
acc = accuracy(y, y_pred)
println("Model accuracy: ", acc)


optimal_weights, optimal_bias, optimal_loss, status = optimize_perceptron_jump(X, y, 1000)

# Create a new Perceptron with the optimal parameters
optimized_perceptron = Perceptron(0.01, size(X, 2), 1)
optimized_perceptron.weights = optimal_weights
optimized_perceptron.bias = optimal_bias

# Make predictions and calculate accuracy
y_pred = predict(optimized_perceptron, X)
acc = accuracy(y, y_pred)
println("Optimized model accuracy: ", acc)
