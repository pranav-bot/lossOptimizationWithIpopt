using Flux
using Zygote
using GaussianProcesses
using KernelFunctions
using Distributions
using Optim
using JuMP
using Ipopt
using LinearAlgebra

# Define the XOR dataset
X = [0 0;
     0 1;
     1 0;
     1 1]
Y = [0.0; 1.0; 1.0; 0.0] # Ensure Y is of type Float64

# Neural network architecture parameters
input_dim = 2
hidden_dim = 2
output_dim = 1
n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim # Total parameters

# Function to create the neural network model with given weights
function create_model(weights)
    W1 = reshape(weights[1:input_dim*hidden_dim], (hidden_dim, input_dim))
    b1 = weights[input_dim*hidden_dim+1 : input_dim*hidden_dim+hidden_dim]
    W2 = reshape(weights[input_dim*hidden_dim+hidden_dim+1 : input_dim*hidden_dim+hidden_dim+hidden_dim*output_dim], (output_dim, hidden_dim))
    b2 = weights[end] # Last parameter
    model(x) = W2 * relu.(W1 * x .+ b1) .+ b2
    return model
end

# Loss function to compute mean squared error
function loss_function(weights)
    model = create_model(weights)
    preds = [model(x)[1] for x in eachrow(X)] # Extract scalar from each prediction
    loss = mean((preds .- Y).^2)
    return loss
end

# Latin Hypercube Sampling to generate initial weights
function latin_hypercube_sampling(n_samples, n_dims)
    points = zeros(n_samples, n_dims)
    for i in 1:n_dims
        perm = randperm(n_samples)
        intervals = range(0, 1; length = n_samples + 1)
        for j in 1:n_samples
            low = intervals[perm[j]]
            high = intervals[perm[j]+1]
            points[j,i] = rand() * (high - low) + low
        end
    end
    return points
end

# Expected Improvement acquisition function
function expected_improvement(x, gp, f_min)
    μ, σ2 = predict_y(gp, reshape(x, 1, :))
    σ = sqrt(σ2[1])
    σ = max(σ, 1e-6) # Ensure standard deviation is positive
    Z = (f_min - μ[1]) / σ
    Φ = cdf(Normal(0,1), Z)
    φ = pdf(Normal(0,1), Z)
    EI = (f_min - μ[1]) * Φ + σ * φ
    return -EI # Negative because we minimize the acquisition function
end

# Initialize parameters
n_initial_samples = 10 # Number of initial samples
n_iterations = 20 # Total iterations

# Step 1: Generate initial samples using Latin Hypercube Sampling
initial_samples = latin_hypercube_sampling(n_initial_samples, n_params)
initial_samples = 2 * initial_samples .- 1 # Scale samples to [-1, 1]

# Lists to store weights, losses, and gradients
weights_list = Vector{Vector{Float64}}()
losses_list = Float64[]
grads_list = Vector{Vector{Float64}}()

# Step 3: Evaluate the loss and gradients for initial samples
for i in 1:n_initial_samples
    weights = initial_samples[i,:]
    weights = collect(weights)
    loss, back = Zygote.pullback(loss_function, weights)
    grad = back(1.0)[1]
    push!(weights_list, weights)
    push!(losses_list, loss)
    push!(grads_list, grad)
end

# Step 7: Iterate the optimization process
for iter in 1:n_iterations
    println("Iteration $iter")

    # Step 4: Update the surrogate model
    X_train = hcat(weights_list...)' # Convert list to matrix (n_samples x n_params)
    Y_train = Float64.(losses_list) # Convert losses to vector

    # Define the Mat52Ard kernel with different length scales for each input dimension
    length_scales = fill(log(1.0), size(X_train, 2)) # Initialize length scales (log scale)
    signal_std = log(1.0) # Initialize signal standard deviation (log scale)
    k = Mat52Ard(length_scales, signal_std)

    # Create the GP regression model
    gp = GP(X_train, Y_train, MeanConst(mean(Y_train)), k)

    # Optimize the GP hyperparameters
    optimize!(gp)

    # Step 5: Define the acquisition function
    f_min = minimum(Y_train)
    acquisition_function(x) = expected_improvement(x, gp, f_min)

    # Step 6: Optimize the acquisition function to find the next sample
    lower_bounds = -ones(n_params)
    upper_bounds = ones(n_params)
    x0 = rand(n_params) .* (upper_bounds .- lower_bounds) .+ lower_bounds # Random initial point
    result = optimize(acquisition_function, lower_bounds, upper_bounds, x0, Fminbox(NelderMead()))

    x_next = result.minimizer

    # Step 3: Evaluate the loss and gradient at the new sample
    x_next = collect(x_next)
    loss, back = Zygote.pullback(loss_function, x_next)
    grad = back(1.0)[1]

    # Append new data to the lists
    push!(weights_list, x_next)
    push!(losses_list, loss)
    push!(grads_list, grad)
end

# After approximating the search space, split it into convex regions and solve each using IPOPT
println("\nStarting IPOPT optimization in convex regions")

n_regions = 5 # Number of regions along each parameter dimension

# List to store the results from IPOPT optimization
ipopt_results = []

for idx in 1:n_regions
    println("Optimizing region $idx")
    # Define bounds for this region
    lb = fill(-1.0 + 2.0*(idx-1)/n_regions, n_params)
    ub = fill(-1.0 + 2.0*idx/n_regions, n_params)

    # Ensure bounds are within [-1, 1]
    lb = clamp.(lb, -1.0, 1.0)
    ub = clamp.(ub, -1.0, 1.0)

    # Random starting point within bounds
    x0 = rand(n_params) .* (ub .- lb) .+ lb

    # Define the GP mean function as the objective
    function gp_mean_objective(x_vals)
        μ, _ = predict_f(gp, reshape(x_vals, 1, :))
        return μ[1]
    end

    # Define the gradient of the GP mean function
    function gp_mean_gradient(x_vals)
        μ_func = x -> (predict_f(gp, reshape(x, 1, :))[1][1])
        grad = Zygote.gradient(μ_func, x_vals)[1]
        return grad
    end

    # Set up JuMP model with Ipopt optimizer
    model = Model(Ipopt.Optimizer)

    @variable(model, lb[i] <= x[i=1:n_params] <= ub[i], start = x0[i])

    function eval_objective(x_vals...)
        x_array = [x_vals...]
        return gp_mean_objective(x_array)
    end

    function eval_gradient(x_vals...)
        x_array = [x_vals...]
        return gp_mean_gradient(x_array)
    end

    register(model, :obj, n_params, eval_objective; autodiff = false)
    register(model, :grad_obj, n_params, eval_gradient; autodiff = false)

    @NLobjective(model, Min, obj(x...))

    optimize!(model)

    x_opt = value.(x)
    f_opt = objective_value(model)

    println("Region $idx optimal GP mean loss: $f_opt")
    println("Region $idx optimal weights: $x_opt")

    # Evaluate the actual loss at the optimal weights
    actual_loss = loss_function(x_opt)
    println("Region $idx actual loss: $actual_loss")

    push!(ipopt_results, (x_opt, actual_loss))
end

# Find and display the best weights and corresponding loss from IPOPT
actual_losses = [res[2] for res in ipopt_results]
idx_best_ipopt = argmin(actual_losses)
best_weights_ipopt = ipopt_results[idx_best_ipopt][1]
best_loss_ipopt = ipopt_results[idx_best_ipopt][2]
println("\nBest loss from IPOPT: $best_loss_ipopt")
println("Best weights from IPOPT: $best_weights_ipopt")

# Evaluate the model with the best weights from IPOPT
best_model_ipopt = create_model(best_weights_ipopt)
preds_ipopt = [best_model_ipopt(x)[1] for x in eachrow(X)]
println("\nPredictions with the best weights from IPOPT: $preds_ipopt")
println("Actual: $Y")
