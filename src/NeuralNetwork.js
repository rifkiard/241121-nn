const Layer = require('./Layer');


class NeuralNetwork {
    layers;
    activationFunctions;

    constructor(neuronCounts, activationFunctions = []) {
        this.activationFunctions = activationFunctions;

        this.layers = new Array(neuronCounts.length - 1);

        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i] = new Layer(neuronCounts[i], neuronCounts[i + 1]);
        }
    }

    feedForward(inputValues) {
        let output = this.layers[0].feedForward(inputValues, this.activationFunctions[0] || null);

        for (let i = 1; i < this.layers.length; i++) {
            output = this.layers[i].feedForward(output, this.activationFunctions[i] || null);
        }

        return output;
    }
}

module.exports = NeuralNetwork;