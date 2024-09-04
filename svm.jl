include("encoders.jl")
using LinearAlgebra
using CSV
using DataFrames
using Statistics
using CategoricalArrays
using Random

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

function accuracy(y_pred::Vector{Float64}, y_true::Vector{Float64})
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

        #println("Epoch $epoch: Loss = $total_loss, w = $(svm.w), b = $(svm.b), Best Loss = $(svm.best_loss)")
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

# Train the multiclass SVM model
classifiers = fit_multiclass_ovr(X_train, y_train, n_classes)

# Predict on the test set
y_pred_test = [Float64(predict_multiclass_ovr(classifiers, X_test[i, :])) for i in 1:size(X_test, 1)]

# Calculate accuracy on the test set
test_accuracy = accuracy(y_pred_test, y_test)
println("Test Accuracy: ", test_accuracy)
