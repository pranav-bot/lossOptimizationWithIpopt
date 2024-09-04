using LinearAlgebra

# Function to split the non-convex search space into convex regions
function split_search_space(bounds::Tuple{Vector{Float64}, Vector{Float64}}, num_regions::Int)
    # Extract lower and upper bounds for each dimension
    lower_bounds, upper_bounds = bounds
    n_dims = length(lower_bounds)
    
    # Compute the number of splits per dimension
    num_splits_per_dim = ceil(Int, num_regions^(1/n_dims))
    
    # Generate the convex regions
    regions = []
    
    # Recursive function to generate regions for each dimension
    function generate_regions(current_region::Vector{Tuple{Float64, Float64}}, dim::Int)
        if dim > n_dims
            push!(regions, current_region)
            return
        end
        
        lower_bound, upper_bound = lower_bounds[dim], upper_bounds[dim]
        split_points = range(lower_bound, upper_bound, length=num_splits_per_dim+1)
        
        for i in 1:num_splits_per_dim
            new_region = copy(current_region)
            new_region[dim] = (split_points[i], split_points[i+1])
            generate_regions(new_region, dim+1)
        end
    end
    
    # Initialize the recursive region generation
    generate_regions(fill((NaN, NaN), n_dims), 1)
    
    return regions
end

# Example usage
bounds = ([0.0, 0.0], [10.0, 10.0])  # Example bounds for a 2D space
num_regions = 16  # Number of desired convex regions

regions = split_search_space(bounds, num_regions)

# Print the regions
for (i, region) in enumerate(regions)
    println("Region $i: ", region)
end
