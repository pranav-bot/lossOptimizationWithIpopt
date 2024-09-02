using Statistics

function MeanSquaredError(y_true, y_pred)
    return mean((y_pred - y_true) .^ 2)
end