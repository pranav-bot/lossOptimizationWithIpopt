using CSV, DataFrames, LinearAlgebra, Random, Plots

# Sigmoid activation function
function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

# Cross-Entropy Loss function
function cross_entropy_loss(y, predictions)
    n_samples = length(y)
    return -mean(y .* log.(predictions) .+ (1 .- y) .* log.(1 .- predictions))
end

# Training function for Perceptron with given initialization
function train_perceptron(X, y, weights_init; epochs=1000, learning_rate=0.01)
    n_samples, n_features = size(X)
    
    # Add bias term to the feature matrix
    X_bias = hcat(ones(n_samples), X)
    
    # Initialize weights (including bias) with the specified method
    weights = weights_init(n_features + 1)
    
    # Store initial weights
    initial_weights = copy(weights)
    
    for epoch in 1:epochs
        # Forward pass: compute predictions
        predictions = sigmoid(X_bias * weights)
        
        # Compute the gradient
        gradients = (X_bias' * (predictions - y)) / n_samples
        
        # Update weights
        weights -= learning_rate * gradients
    end
    
    # Extract bias and feature weights
    bias = weights[1]
    feature_weights = weights[2:end]
    
    # Print initial and final weights and biases
    println("Initial Weights: ", initial_weights)
    println("Final Weights: ", weights)
    println("Bias: ", bias)
    println("Feature Weights: ", feature_weights)
    
    return bias, feature_weights
end

# Perceptron training function with zero initialization
function weights_zero_init(n_weights)
    return zeros(n_weights)
end

# Perceptron training function with random initialization
function weights_random_init(n_weights)
    return randn(n_weights)
end

# Function to compute the loss surface
function compute_loss_surface(X, y, weight_ranges; weights_init)
    n_samples, n_features = size(X)
    X_bias = hcat(ones(n_samples), X)
    
    loss_surface = zeros(length(weight_ranges[1]), length(weight_ranges[2]))
    
    for (i, w1) in enumerate(weight_ranges[1])
        for (j, w2) in enumerate(weight_ranges[2])
            weights = weights_init(n_features + 1)
            weights[2] = w1
            weights[3] = w2
            
            predictions = sigmoid(X_bias * weights)
            loss_surface[i, j] = cross_entropy_loss(y, predictions)
        end
    end
    
    return loss_surface
end

# Main function to run the experiments
function main()
    # Load and prepare the Iris dataset
    df = CSV.read("data/Iris.csv", DataFrame)
    
    # Convert categorical feature to integer values
    X = Matrix(df[!, Not(:Species)])  # Features
    y_raw = df.Species

    # Create a dictionary for encoding
    label_dict = Dict("Iris-setosa" => 1, "Iris-versicolor" => 2, "Iris-virginica" => 3)
    y_encoded = [label_dict[label] for label in y_raw]

    # One-hot encode the target variable
    n_classes = 3
    y_one_hot = hcat([y_encoded .== i for i in 1:n_classes]...)

    # Weight ranges for the plot
    weight_ranges = (collect(-5:0.5:5), collect(-5:0.5:5))

    # Train Perceptron with zero initialization and compute loss surface
    println("Training Perceptron with Zero Initialization:")
    bias_zero, feature_weights_zero = train_perceptron(X, y_one_hot[:, 1], weights_zero_init; epochs=1000, learning_rate=0.01)
    
    println("\nComputing loss surface for Zero Initialization:")
    loss_surface_zero = compute_loss_surface(X, y_one_hot[:, 1], weight_ranges; weights_init=weights_zero_init)
    
    # Train Perceptron with random initialization and compute loss surface
    println("\nTraining Perceptron with Random Initialization:")
    bias_random, feature_weights_random = train_perceptron(X, y_one_hot[:, 1], weights_random_init; epochs=1000, learning_rate=0.01)
    
    println("\nComputing loss surface for Random Initialization:")
    loss_surface_random = compute_loss_surface(X, y_one_hot[:, 1], weight_ranges; weights_init=weights_random_init)

    # Plot the loss surfaces
    p1 = contour(weight_ranges[1], weight_ranges[2], loss_surface_zero, title="Loss Surface (Zero Initialization)", xlabel="Weight 1", ylabel="Weight 2")
    p2 = contour(weight_ranges[1], weight_ranges[2], loss_surface_random, title="Loss Surface (Random Initialization)", xlabel="Weight 1", ylabel="Weight 2")

    # Display plots
    plot(p1, p2, layout=(1, 2))
end

# Run the main function
main()
