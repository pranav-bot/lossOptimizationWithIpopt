using Pkg
using Flux
using LatinHypercubeSampling
using Statistics  # For calculating means and standard deviations

# Define XOR dataset
function xor_dataset()
    X = Float32[0 0; 0 1; 1 0; 1 1]'  # Shape (4, 2)
    y = Float32[0; 1; 1; 0]            # Shape (4,)
    return X, y
end

# Generate weight and bias samples using LHS
function generate_samples(num_samples, num_weights, num_biases, gens)
    weight_plan, _ = LHCoptim(num_samples, num_weights, gens)
    bias_plan, _ = LHCoptim(num_samples, num_biases, gens)
    weights = (weight_plan .- 0.5) * 4  # Scale weights to [-2, 2]
    biases = (bias_plan .- 0.5) * 4      # Scale biases to [-2, 2]
    return weights, biases
end

# Create model
input_dim = 2
hidden_dim = 2
output_dim = 1
num_weights = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim)
num_biases = hidden_dim + output_dim

function create_model()
    return Chain(
        Dense(input_dim, hidden_dim, relu),  # Hidden layer with 2 neurons and ReLU activation
        Dense(hidden_dim, output_dim, σ)      # Output layer with sigmoid activation
    )
end

# Function to evaluate model and calculate loss
function evaluate_loss(model, X, y)
    loss_function = Flux.Losses.binarycrossentropy
    y_hat = model(X)  # Get predictions
    y_hat = reshape(y_hat, size(y))
    return loss_function(y_hat, y)
end

function set_parameters(model, weights, biases)
    num_hidden_weights = input_dim * hidden_dim
    num_hidden_biases = hidden_dim
    num_output_weights = hidden_dim * output_dim
    num_output_biases = output_dim

    # Set weights and biases for the first layer
    model[1].weight .= reshape(weights[1:num_hidden_weights], hidden_dim, input_dim)
    model[1].bias .= biases[1:num_hidden_biases]

    # Set weights and biases for the second layer
    start_idx = num_hidden_weights + num_hidden_biases + 1
    end_idx = start_idx + num_output_weights - 1
    model[2].weight .= reshape(weights[start_idx:end_idx], output_dim, hidden_dim)
    model[2].bias .= biases[num_hidden_biases + 1:end]  # Correct indexing for biases
end

num_samples = 100

# Generate initial weight and bias samples
weights_samples, biases_samples = generate_samples(num_samples, num_weights, num_biases, 1000)
println("Weight Samples: ")
println(weights_samples)
println("Bias Samples: ")
println(biases_samples)

model = create_model()

# Get the XOR inputs
X, y = xor_dataset()

outputs = []

# Iterate through each weight and bias sample
for i in 1:num_samples
    set_parameters(model, weights_samples[i, :], biases_samples[i, :])
    push!(outputs, model(X))
end

# Convert outputs to a more manageable format (e.g., a matrix)
outputs_matrix = hcat(outputs...)

println("Outputs for each weight sample: ")
println(outputs_matrix)

# Evaluate loss for each weight and bias sample
losses = Float32[]

for i in 1:num_samples
    set_parameters(model, weights_samples[i, :], biases_samples[i, :])
    loss = evaluate_loss(model, X, y)
    push!(losses, loss)
end

# Analyze loss values
mean_loss = mean(losses)
std_loss = std(losses)

println("Mean Loss: $mean_loss")
println("Standard Deviation of Loss: $std_loss")

# Identify promising regions
lower_loss_indices = findall(losses .< mean_loss)  # Indices of weights with loss lower than mean
promising_weights = weights_samples[lower_loss_indices, :]
promising_biases = biases_samples[lower_loss_indices, :]

println("Promising Weights (Lower Loss):")
println(promising_weights)
println("Corresponding Biases (Lower Loss):")
println(promising_biases)