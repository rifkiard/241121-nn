

const sigmoid = {
    name: 'sigmoid',
    fn: x => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
        const fx = sigmoid.fn(x);
        return fx * (1 - fx);
    }
}

const reLU = {
    name: 'reLU',
    fn: x => Math.max(0, x),
    derivative: x => x > 0 ? 1 : 0
}

const softmax = {
    name: 'softmax',
    fn: (logits) => {
        const maxLogit = logits.reduce((a, b) => Math.max(a, b), -Infinity);
        const scores = logits.map((l) => Math.exp(l - maxLogit));
        const denom = scores.reduce((a, b) => a + b);
        return scores.map((s) => s / denom);
    },
    derivative: (outputs, targets) => {
        // const probabilities = softmax.fn(logits);
        // const jacobian = [];

        // for (let i = 0; i < probabilities.length; i++) {
        //     jacobian[i] = [];
        //     for (let j = 0; j < probabilities.length; j++) {
        //         if (i === j) {
        //             jacobian[i][j] = probabilities[i] * (1 - probabilities[i]);
        //         } else {
        //             jacobian[i][j] = -probabilities[i] * probabilities[j];
        //         }
        //     }
        // }

        // return jacobian;

        // Simplified gradient for softmax with cross-entropy
        return outputs.map((output, i) => output - targets[i]);
    }
}

module.exports = {
    sigmoid,
    reLU,
    softmax
};
