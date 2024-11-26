

const sigmoid = {
    name: 'sigmoid',
    fn: x => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
        const fx = sigmoid.fn(x);
        return fx * (1 - fx);
    }
}

module.exports = {
    sigmoid,
};
