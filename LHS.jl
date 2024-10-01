using JuMP
using Ipopt
using LatinHypercubeSampling
using Plots
using Statistics
using Clustering

# XOR dataset
xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

# Network parameters
input_dim = 2
hidden_dim = 2
output_dim = 1
num_weights = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim

# Activation function
activation_function(x) = tanh(x)

# Function to generate weight samples using LHS
function generate_weight_samples(num_samples, num_weights, gens=1000)
    plan, _ = LHCoptim(num_samples, num_weights, gens)
    return (plan .- 0.5) * 4  # Scale to [-2, 2] range
end

# Function to split weights into network parameters
function split_weights(weights)
    W1 = reshape(weights[1:(input_dim * hidden_dim)], (hidden_dim, input_dim))
    b1 = weights[(input_dim * hidden_dim + 1):(input_dim * hidden_dim + hidden_dim)]
    W2 = reshape(weights[(input_dim * hidden_dim + hidden_dim + 1):(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim)], (output_dim, hidden_dim))
    b2 = weights[(end - output_dim + 1):end]
    return W1, b1, W2, b2
end

# Function to compute loss for a given set of weights
function compute_loss(weights)
    W1, b1, W2, b2 = split_weights(weights)
    total_loss = 0.0
    for (x1, x2, y) in xor_data
        h = activation_function.(W1 * [x1, x2] .+ b1)
        y_pred = activation_function(W2 * h .+ b2)[1]
        total_loss += (y_pred - y)^2
    end
    return total_loss / length(xor_data)
end

# Function to compute gradients for a given set of weights
function compute_gradients(weights)
    W1, b1, W2, b2 = split_weights(weights)
    gradients = zeros(length(weights))
    for (x1, x2, y) in xor_data
        h = activation_function.(W1 * [x1, x2] .+ b1)
        y_pred = activation_function(W2 * h .+ b2)[1]
        
        # Backpropagation
        delta_output = 2 * (y_pred - y) * (1 - y_pred^2)
        delta_hidden = (W2' * [delta_output]) .* (1 .- h.^2)
        
        # Gradients for W2 and b2
        gradients[(input_dim * hidden_dim + hidden_dim + 1):(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim)] .+= vec(delta_output * h')
        gradients[(end - output_dim + 1):end] .+= delta_output
        
        # Gradients for W1 and b1
        gradients[1:(input_dim * hidden_dim)] .+= vec(delta_hidden * [x1, x2]')
        gradients[(input_dim * hidden_dim + 1):(input_dim * hidden_dim + hidden_dim)] .+= delta_hidden
    end
    return gradients / length(xor_data)
end

# Function to analyze loss landscape and split into convex regions
function analyze_and_split_loss_landscape(weight_samples, num_regions)
    losses = [compute_loss(weight_samples[i, :]) for i in 1:size(weight_samples, 1)]
    gradients = [compute_gradients(weight_samples[i, :]) for i in 1:size(weight_samples, 1)]
    
    # Compute gradient magnitudes
    grad_mags = [norm(g) for g in gradients]
    
    # Combine loss and gradient information
    landscape_features = hcat(losses, grad_mags)
    
    # Normalize features
    normalized_features = (landscape_features .- minimum(landscape_features, dims=1)) ./ 
                          (maximum(landscape_features, dims=1) .- minimum(landscape_features, dims=1))
    
    # Perform k-means clustering
    kmeans_result = kmeans(normalized_features', num_regions)
    
    # Split the weight samples into convex regions based on the clustering
    convex_regions = [weight_samples[kmeans_result.assignments .== i, :] for i in 1:num_regions]
    
    return convex_regions, losses, grad_mags
end

# Function to solve XOR problem for a convex region using IPOPT
function solve_convex_region(convex_region)
    model = Model(Ipopt.Optimizer)
    
    @variable(model, w[1:num_weights])
    
    # Set initial point as the mean of the convex region
    set_start_value.(w, vec(mean(convex_region, dims=1)))
    
    # Objective function
    @NLobjective(model, Min, compute_loss(w))
    
    optimize!(model)
    
    return objective_value(model), value.(w)
end

# Main workflow
num_samples = 10000
num_regions = 5

# Generate weight samples using LHS
weight_samples = generate_weight_samples(num_samples, num_weights)

# Analyze loss landscape and split into convex regions
convex_regions, losses, grad_mags = analyze_and_split_loss_landscape(weight_samples, num_regions)

# Solve each convex region
best_loss = Inf
best_weights = nothing

for (i, region) in enumerate(convex_regions)
    println("Solving region $i")
    loss, weights = solve_convex_region(region)
    if loss < best_loss
        best_loss = loss
        best_weights = weights
    end
end

println("Best loss: ", best_loss)

# Visualize the decision boundary
W1, b1, W2, b2 = split_weights(best_weights)

x1_range = range(-0.1, 1.1, length=100)
x2_range = range(-0.1, 1.1, length=100)
z = [activation_function(W2 * activation_function.(W1 * [x1, x2] .+ b1) .+ b2)[1] for x1 in x1_range, x2 in x2_range]

p1 = contour(x1_range, x2_range, z, levels=20, fill=true, title="XOR Decision Boundary")
scatter!(p1, [point[1] for point in xor_data], [point[2] for point in xor_data], 
         marker_z=[point[3] for point in xor_data], color=:viridis, 
         markersize=6, legend=false)

# Visualize the loss landscape
p2 = scatter(losses, grad_mags, xlabel="Loss", ylabel="Gradient Magnitude", 
             title="Loss Landscape", legend=false, alpha=0.5)

# Combine plots
plot(p1, p2, layout=(1, 2), size=(1000, 400))

savefig("xor_analysis_lhs.png")
println("XOR analysis plot saved as 'xor_analysis_lhs.png'")