using Plots
using Statistics

function plot_loss_surface(x, y, b_history, m_history)
    # Define the range for b and m based on historical values
    b_min, b_max = minimum(b_history), maximum(b_history)
    m_min, m_max = minimum(m_history), maximum(m_history)
    
    b_values = range(b_min, stop=b_max, length=100)
    m_values = range(m_min, stop=m_max, length=100)
    losses = Matrix{Float64}(undef, length(b_values), length(m_values))
    
    for (i, b) in enumerate(b_values)
        for (j, m) in enumerate(m_values)
            ŷ = b .+ m .* x
            losses[i, j] = meanSquaredError(y, ŷ)
        end
    end

    # Create 3D surface plot
    fig = surface(
        b_values, m_values, losses,
        xlabel = "Intercept (b)",
        ylabel = "Slope (m)",
        zlabel = "Loss",
        title = "Loss Surface with Respect to Intercept and Slope"
    )

    # Save plot to a file
    savefig("./plots/loss_surface.png")
    println("Plot saved as loss_surface.png")
end