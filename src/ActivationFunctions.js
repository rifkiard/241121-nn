
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function relu(x) {
    return Math.max(0, x);
}

module.exports = {
    sigmoid,
    relu
};