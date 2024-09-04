include("encoders.jl")
using LinearAlgebra
using CSV
using DataFrames
using Statistics
using Random

# SVM with Gradient Descent
mutable struct SVM
    w::Vector{Float64}
    b::Float64
    learning_rate::Float64
    C::Float64
    w_history::Vector{Vector{Float64}}
    b_history::Vector{Float64}
    loss_history::Vector{Float64}
    best_w::Vector{Float64}
    best_b::Float64
    best_loss::Float64
end

function SVM_zero_init(n_features::Int, learning_rate::Float64=0.01, C::Float64=1.0)
    return SVM(zeros(n_features), 0.0, learning_rate, C, Vector{Vector{Float64}}(), Float64[], Float64[], zeros(n_features), 0.0, Inf)
end

function SVM_random_init(n_features::Int, learning_rate::Float64=0.01, C::Float64=1.0)
    return SVM(rand(n_features), randn(), learning_rate, C, Vector{Vector{Float64}}(), Float64[], Float64[], rand(n_features), randn(), Inf)
end

function predict_classifier(svm::SVM, X::Vector{Float64})
    return dot(svm.w, X) + svm.b >= 0 ? 1 : -1
end

function predict_regressor(svm::SVM, X::Vector{Float64})
    return dot(svm.w, X) + svm.b
end

function hinge_loss_classification(svm::SVM, X::Vector{Float64}, y::Float64)
    return max(0, 1 - y * (predict_regressor(svm, X)))
end

function accuracy(y_pred::Vector{Int64}, y_true::Vector{Float64})
    return mean(y_pred .== y_true)
end

function fit_classification!(svm::SVM, X::Matrix{Float64}, y::Vector{Float64}, epochs::Int=100)
    n_samples, n_features = size(X)

    for epoch in 1:epochs
        total_loss = 0.0

        for i in 1:n_samples
            if y[i] * predict_regressor(svm, X[i, :]) < 1
                svm.w .= svm.w .- svm.learning_rate * (svm.w - svm.C * y[i] * X[i, :])
                svm.b -= svm.learning_rate * y[i]
            else
                svm.w .= svm.w .- svm.learning_rate * svm.w
            end
            total_loss += hinge_loss_classification(svm, X[i, :], y[i])
        end

        # Store weights, bias, and loss
        push!(svm.w_history, copy(svm.w))
        push!(svm.b_history, svm.b)
        push!(svm.loss_history, total_loss)

        # Update best loss
        if total_loss < svm.best_loss
            svm.best_loss = total_loss
            svm.best_w .= svm.w
            svm.best_b = svm.b
        end
    end

    println("Final weights: ", svm.w)
    println("Final bias: ", svm.b)
    println("Best weights: ", svm.best_w)
    println("Best bias: ", svm.best_b)
    println("Best loss: ", svm.best_loss)
end

function fit_multiclass_ovr(X::Matrix{Float64}, y::Vector{Float64}, n_classes::Int, epochs::Int=1000, learning_rate::Float64=0.000001, C::Float64=1000.0)
    n_samples, n_features = size(X)
    classifiers = [SVM_zero_init(n_features, learning_rate, C) for _ in 1:n_classes]

    for class in 1:n_classes
        y_binary = map(y_i -> y_i == class ? 1.0 : -1.0, y)
        fit_classification!(classifiers[class], X, y_binary, epochs)
    end
    return classifiers
end

function predict_multiclass_ovr(classifiers::Vector{SVM}, X::Vector{Float64})
    scores = [predict_regressor(classifier, X) for classifier in classifiers]
    return argmax(scores)
end

# SMO Algorithm Implementation
mutable struct SMO
    alphas::Vector{Float64}
    b::Float64
    C::Float64
    tol::Float64
    kernel::Function
    X::Matrix{Float64}
    y::Vector{Float64}
end

function SMO_init(X::Matrix{Float64}, y::Vector{Float64}, C::Float64=1.0, tol::Float64=1e-3, kernel::Function=dot)
    n_samples, _ = size(X)
    return SMO(zeros(n_samples), 0.0, C, tol, kernel, X, y)
end

function predict_regressor_smo(smo::SMO, X::Vector{Float64})
    return sum(smo.alphas .* smo.y .* [smo.kernel(X, smo.X[i, :]) for i in 1:length(smo.alphas)]) + smo.b
end

function smo_train!(smo::SMO, epochs::Int=100)
    n_samples, _ = size(smo.X)

    for epoch in 1:epochs
        alpha_prev = copy(smo.alphas)
        
        for i in 1:n_samples
            xi = smo.X[i, :]
            yi = smo.y[i]
            Ei = predict_regressor_smo(smo, xi) - yi
            
            if (yi * Ei < -smo.tol && smo.alphas[i] < smo.C) || (yi * Ei > smo.tol && smo.alphas[i] > 0)
                j = rand(1:n_samples)
                xj = smo.X[j, :]
                yj = smo.y[j]
                Ej = predict_regressor_smo(smo, xj) - yj
                
                alpha_i_old = smo.alphas[i]
                alpha_j_old = smo.alphas[j]
                
                if yi != yj
                    L = max(0.0, alpha_j_old - alpha_i_old)
                    H = min(smo.C, smo.C + alpha_j_old - alpha_i_old)
                else
                    L = max(0.0, alpha_i_old + alpha_j_old - smo.C)
                    H = min(smo.C, alpha_i_old + alpha_j_old)
                end
                
                if L == H
                    continue
                end
                
                eta = 2 * smo.kernel(xi, xj) - smo.kernel(xi, xi) - smo.kernel(xj, xj)
                if eta >= 0
                    continue
                end
                
                smo.alphas[j] -= yj * (Ei - Ej) / eta
                smo.alphas[j] = clamp(smo.alphas[j], L, H)
                
                if abs(smo.alphas[j] - alpha_j_old) < smo.tol
                    continue
                end
                
                smo.alphas[i] += yi * yj * (alpha_i_old - smo.alphas[j])
                
                b1 = smo.b - Ei - yi * (smo.alphas[i] - alpha_i_old) * smo.kernel(xi, xi) - yj * (smo.alphas[j] - alpha_j_old) * smo.kernel(xi, xj)
                b2 = smo.b - Ej - yi * (smo.alphas[i] - alpha_i_old) * smo.kernel(xi, xj) - yj * (smo.alphas[j] - alpha_j_old) * smo.kernel(xj, xj)
                
                if 0 < smo.alphas[i] < smo.C
                    smo.b = b1
                elseif 0 < smo.alphas[j] < smo.C
                    smo.b = b2
                else
                    smo.b = (b1 + b2) / 2
                end
            end
        end
        
        if maximum(abs.(smo.alphas .- alpha_prev)) < smo.tol
            break
        end
    end
end

function split_dataset(X::Matrix{Float64}, y::Vector{Float64}, ratio::Float64=0.7)
    n_samples = size(X, 1)
    indices = 1:n_samples
    shuffled_indices = shuffle(indices)
    split_index = Int(round(ratio * n_samples))

    X_train = X[shuffled_indices[1:split_index], :]
    y_train = y[shuffled_indices[1:split_index]]
    X_test = X[shuffled_indices[split_index+1:end], :]
    y_test = y[shuffled_indices[split_index+1:end]]

    return (X_train, y_train, X_test, y_test)
end

# Load and prepare data
data = CSV.File("data/Iris.csv")
y = encodemake(Vector(data.Species))

data = DataFrame(data)
select!(data, Not(names(data)[end]))
select!(data, Not(names(data)[1]))
x = Matrix(data)

n_classes = length(unique(y))

# Split the dataset
(X_train, y_train, X_test, y_test) = split_dataset(x, y)

# Train the multiclass SVM model with GD
classifiers_gd = fit_multiclass_ovr(X_train, y_train, n_classes)

# Train the multiclass SVM model with SMO
classifiers_smo = [SMO_init(X_train, y_train, 1.0, 1e-3) for _ in 1:n_classes]
for classifier in classifiers_smo
    smo_train!(classifier)
end

# Prediction with GD
y_pred_test_gd = [predict_multiclass_ovr(classifiers_gd, X_test[i, :]) for i in 1:size(X_test, 1)]
test_accuracy_gd = accuracy(y_pred_test_gd, y_test)
println("Test Accuracy with GD: ", test_accuracy_gd)

# Prediction with SMO
y_pred_test_smo = [predict_regressor_smo(classifiers_smo[1], X_test[i, :]) for i in 1:size(X_test, 1)]
y_pred_test_smo = [argmax([predict_regressor_smo(classifiers_smo[c], X_test[i, :]) for c in 1:n_classes]) for i in 1:size(X_test, 1)]
test_accuracy_smo = accuracy(y_pred_test_smo, y_test)
println("Test Accuracy with SMO: ", test_accuracy_smo)
