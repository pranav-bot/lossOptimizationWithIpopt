using Statistics

function meanSquaredError(y, ŷ)
    return mean((y - ŷ) .^ 2)
end

function binaryCrossEntropy(y, ŷ; ϵ=1e-15)
    ŷ = clamp.(ŷ, ϵ, 1-ϵ)
    return -mean(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ))
end

function binaryCrossEntropy(y::Vector{Float64}, ŷ::Vector{NonlinearExpr}; ϵ::Float64=1e-10)
    return -sum(y .* log.(ŷ .+ ϵ) .+ (1 .- y) .* log.(1 .- ŷ .+ ϵ))
end

function scalarBinaryCrossEntropy(y::Float64, ŷ::NonlinearExpr; ϵ::Float64=1e-10)
    return -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)
end

function scalarBinaryCrossEntropy(y::T, ŷ::T; ϵ::T=1e-10) where {T<:Real}
    return -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)
end


a = [1,0,1,1,1]
b = [0,0,0,0,0]

print(binaryCrossEntropy(a, b))