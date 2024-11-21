class Layer {
    inputNeuronCount;
    outputNeuronCount;

    inputs;
    outputs;
    biases;
    weights;

    constructor(inputNeuronCount, outputNeuronCount) {
        this.inputNeuronCount = inputNeuronCount;
        this.outputNeuronCount = outputNeuronCount;

        this.inputs = new Array(inputNeuronCount);
        this.outputs = new Array(outputNeuronCount);
        this.biases = new Array(outputNeuronCount);

        this.weights = new Array(outputNeuronCount).fill(new Array(inputNeuronCount));

        this.#initWeights();
        this.#initBiases();
    }

    #initWeights() {
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    #initBiases() {
        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.biases[i] = Math.random() * 2 - 1;
        }
    }

    #computeNeuronOutput(inputValues, weights, bias) {
        let sum = 0;

        for (let i = 0; i < inputValues.length; i++) {
            sum += inputValues[i] * weights[i];
        }

        return sum + bias;
    }

    feedForward(inputValues, activationFunction = null) {
        for (let i = 0; i < this.inputNeuronCount; i++) {
            this.inputs[i] = inputValues[i];
        }

        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.outputs[i] = this.#computeNeuronOutput(this.inputs, this.weights[i], this.biases[i]);
        }

        if (activationFunction !== null) {
            for (let i = 0; i < this.outputNeuronCount; i++) {
                this.outputs[i] = activationFunction(this.outputs[i]);
            }
        }

        return this.outputs;
    }
}

module.exports = Layer;