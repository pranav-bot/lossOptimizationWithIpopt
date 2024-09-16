using Random
using LinearAlgebra
using Plots
using JuMP
using Ipopt

# Define the neural network structure
struct NeuralNetwork
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

# Initialize the neural network
function init_network(input_size::Int, hidden_size::Int, output_size::Int)
    W1 = randn(hidden_size, input_size) * sqrt(2 / input_size)
    b1 = zeros(hidden_size)
    W2 = randn(output_size, hidden_size) * sqrt(2 / hidden_size)
    b2 = zeros(output_size)
    return NeuralNetwork(W1, b1, W2, b2)
end

# Activation function (ReLU)
relu(x) = max.(0, x)

# Forward pass
function forward(nn::NeuralNetwork, X::Matrix{Float64})
    h = relu.(nn.W1 * X .+ nn.b1)
    y = nn.W2 * h .+ nn.b2
    return y
end

# Loss function (Mean Squared Error)
mse_loss(y_pred::Matrix{Float64}, y_true::Matrix{Float64}) = sum((y_pred .- y_true).^2) / size(y_true, 2)

# Generate synthetic data
function generate_data(n_samples::Int)
    X = randn(2, n_samples)
    y = sin.(X[1, :]) .* cos.(X[2, :])
    return X, reshape(y, 1, :)  # Ensure y is a 1xn_samples matrix
end

# Train using SGD
function train_sgd!(nn::NeuralNetwork, X::Matrix{Float64}, y::Matrix{Float64}, lr::Float64, epochs::Int)
    n_samples = size(X, 2)
    losses = Float64[]
    
    for epoch in 1:epochs
        # Forward pass
        h = relu.(nn.W1 * X .+ nn.b1)
        y_pred = nn.W2 * h .+ nn.b2
        
        # Compute loss
        loss = mse_loss(y_pred, y)
        push!(losses, loss)
        
        # Backward pass
        dy = 2 .* (y_pred .- y) ./ n_samples
        dW2 = dy * h'
        db2 = sum(dy, dims=2)
        dh = nn.W2' * dy
        dh[h .<= 0] .= 0
        dW1 = dh * X'
        db1 = sum(dh, dims=2)
        
        # Update parameters
        nn.W1 .-= lr .* dW1
        nn.b1 .-= lr .* vec(db1)
        nn.W2 .-= lr .* dW2
        nn.b2 .-= lr .* vec(db2)
    end
    
    return losses
end

# Optimize using Ipopt
function optimize_ipopt(X::Matrix{Float64}, y::Matrix{Float64}, hidden_size::Int)
    input_size, n_samples = size(X)
    output_size = size(y, 1)
    
    model = Model(Ipopt.Optimizer)
    
    @variable(model, W1[1:hidden_size, 1:input_size])
    @variable(model, b1[1:hidden_size])
    @variable(model, W2[1:output_size, 1:hidden_size])
    @variable(model, b2[1:output_size])
    
    @variable(model, h[1:hidden_size, 1:n_samples])
    @variable(model, y_pred[1:output_size, 1:n_samples])
    
    @constraint(model, [i=1:hidden_size, j=1:n_samples], h[i,j] >= W1[i,:]' * X[:,j] + b1[i])
    @constraint(model, [i=1:hidden_size, j=1:n_samples], h[i,j] >= 0)
    @constraint(model, [i=1:hidden_size, j=1:n_samples], h[i,j] * (h[i,j] - (W1[i,:]' * X[:,j] + b1[i])) <= 0)
    
    @constraint(model, [i=1:output_size, j=1:n_samples], y_pred[i,j] == W2[i,:]' * h[:,j] + b2[i])
    
    @objective(model, Min, sum((y_pred[i,j] - y[i,j])^2 for i in 1:output_size, j in 1:n_samples) / n_samples)
    
    optimize!(model)
    
    return value.(W1), value.(b1), value.(W2), value.(b2)
end

# Main
function main()
    # Generate data
    X, y = generate_data(1000)
    train_size = 800
    X_train, y_train = X[:, 1:train_size], y[:, 1:train_size]
    X_test, y_test = X[:, (train_size+1):end], y[:, (train_size+1):end]
    
    # Initialize network
    input_size, hidden_size, output_size = 2, 10, 1
    nn_sgd = init_network(input_size, hidden_size, output_size)
    
    # Train using SGD
    sgd_losses = train_sgd!(nn_sgd, X_train, y_train, 0.01, 1000)
    
    # Optimize using Ipopt
    W1, b1, W2, b2 = optimize_ipopt(X_train, y_train, hidden_size)
    nn_ipopt = NeuralNetwork(W1, vec(b1), W2, vec(b2))
    
    # Evaluate both models
    y_pred_sgd = forward(nn_sgd, X_test)
    y_pred_ipopt = forward(nn_ipopt, X_test)
    
    loss_sgd = mse_loss(y_pred_sgd, y_test)
    loss_ipopt = mse_loss(y_pred_ipopt, y_test)
    
    println("SGD Loss: ", loss_sgd)
    println("Ipopt Loss: ", loss_ipopt)
    
    # Plot loss surface
    p = plot(1:length(sgd_losses), sgd_losses, label="SGD", xlabel="Epochs", ylabel="Loss", title="Loss Surface")
    hline!([loss_ipopt], label="Ipopt", linestyle=:dash)
    display(p)
    savefig(p, "loss_surface.png")
end

main()