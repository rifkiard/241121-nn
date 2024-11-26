class Layer {
    inputNeuronCount;
    outputNeuronCount;

    inputs;
    outputs;
    biases;
    weights;

    // For back propagation.
    weightGradients;
    biasGradients;
    outputsBeforeActivation;

    constructor(inputNeuronCount, outputNeuronCount) {
        this.inputNeuronCount = inputNeuronCount;
        this.outputNeuronCount = outputNeuronCount;

        this.inputs = new Array(inputNeuronCount);

        this.outputs = new Array(outputNeuronCount);
        this.outputsBeforeActivation = new Array(outputNeuronCount);

        this.biases = new Array(outputNeuronCount);
        this.biasGradients = new Array(outputNeuronCount);

        this.weights = [];

        for (let i = 0; i < outputNeuronCount; i++) {
            this.weights[i] = [];

            for (let j = 0; j < inputNeuronCount; j++) {
                this.weights[i][j] = 0;
            }
        }

        this.weightGradients = [];

        for (let i = 0; i < outputNeuronCount; i++) {
            this.weightGradients[i] = [];

            for (let j = 0; j < inputNeuronCount; j++) {
                this.weightGradients[i][j] = 0;
            }
        }

        this.#randomizeWeights();
        this.#randomizeBiases();
    }

    #randomizeWeights() {
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    #randomizeBiases() {
        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.biases[i] = Math.random() * 2 - 1;
        }
    }

    #computeWeightedSum(inputValues, weights, bias) {
        let sum = 0;

        // dot product.
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
            this.outputsBeforeActivation[i] = this.#computeWeightedSum(this.inputs, this.weights[i], this.biases[i]);
        }

        this.outputs = [...this.outputsBeforeActivation];

        if (activationFunction !== null) {
            for (let i = 0; i < this.outputNeuronCount; i++) {
                this.outputs[i] = activationFunction.fn(this.outputs[i]);
            }
        }

        return this.outputs;
    }

    #updateWeightsAndBiases(learningRate) {
        for (let i = 0; i < this.outputNeuronCount; i++) {
            for (let j = 0; j < this.inputNeuronCount; j++) {
                this.weights[i][j] -= this.weightGradients[i][j] * learningRate;
            }

            this.biases[i] -= this.biasGradients[i] * learningRate;
        }
    }

    backPropagate(
        nextLayerDeltas,
        nextLayerWeights,
        activationFunction,
        learningRate,
        isOutputLayer = false,
        targets = []
    ) {
        const deltas = new Array(this.outputNeuronCount);

        if (isOutputLayer) {
            for (let i = 0; i < this.outputNeuronCount; i++) {
                const error = this.outputs[i] - targets[i];
                deltas[i] = error * activationFunction.derivative(this.outputsBeforeActivation[i]);
            }
        } else {
            for (let i = 0; i < this.outputNeuronCount; i++) {
                let error = 0;

                for (let j = 0; j < nextLayerDeltas.length; j++) {
                    error += nextLayerDeltas[j] * nextLayerWeights[j][i];
                }

                deltas[i] = error * activationFunction.derivative(this.outputsBeforeActivation[i]);
            }
        }

        // calc. gradients.
        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.biasGradients[i] = deltas[i];

            for (let j = 0; j < this.inputNeuronCount; j++) {
                this.weightGradients[i][j] = deltas[i] * this.inputs[j];
            }
        }

        this.#updateWeightsAndBiases(learningRate);


        return deltas;
    }
}

module.exports = Layer;