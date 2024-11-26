class Layer {
    inputNeuronCount;
    outputNeuronCount;

    inputs;
    outputs;
    biases;
    weights;

    currentActivationFunction;
    nextActivationFunction;


    // For back propagation.
    weightGradients;
    biasGradients;
    outputsBeforeActivation;


    constructor(inputNeuronCount, outputNeuronCount, currentActivationFunction, nextActivationFunction) {
        this.currentActivationFunction = currentActivationFunction;
        this.nextActivationFunction = nextActivationFunction;

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
            const inputSize = this.inputNeuronCount;
            const outputSize = this.outputNeuronCount;
            let stddev;

            if (this.currentActivationFunction && this.currentActivationFunction.name === 'reLU') {
                // He initialization
                stddev = Math.sqrt(2.0 / inputSize);
            } else if (this.currentActivationFunction && this.currentActivationFunction.name === 'sigmoid' || this.currentActivationFunction && this.currentActivationFunction.name === 'tanh') {
                // Xavier initialization
                stddev = Math.sqrt(2.0 / (inputSize + outputSize));
            } else {
                // Default to small random weights
                stddev = 0.01;
            }

            for (let j = 0; j < this.weights[i].length; j++) {
                const u1 = 1 - Math.random();
                const u2 = Math.random();
                const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

                this.weights[i][j] = stddev * randStdNormal;
            }
        }
    }

    #randomizeBiases() {
        if (this.currentActivationFunction && this.currentActivationFunction.name === 'reLU') {
            // Biases in the range [-0.5, 0.5] for ReLU
            for (let i = 0; i < this.outputNeuronCount; i++) {
                this.biases[i] = Math.random() * 1 - 0.5;
            }
            return;
        }

        if (this.currentActivationFunction && this.currentActivationFunction.name === 'softmax') {
            // Biases 0 for softmax
            for (let i = 0; i < this.outputNeuronCount; i++) {
                this.biases[i] = 0;
            }
            return;
        }

        // Biases in a small range [-0.05, 0.05] for other activations (e.g., sigmoid, tanh, softmax)
        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.biases[i] = Math.random() * 0.1 - 0.05;
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

    feedForward(inputValues) {
        for (let i = 0; i < this.inputNeuronCount; i++) {
            this.inputs[i] = inputValues[i];
        }

        for (let i = 0; i < this.outputNeuronCount; i++) {
            this.outputsBeforeActivation[i] = this.#computeWeightedSum(this.inputs, this.weights[i], this.biases[i]);
        }

        this.outputs = [...this.outputsBeforeActivation];

        if (this.nextActivationFunction !== null) {
            if (this.nextActivationFunction.name === 'softmax') {
                this.outputs = this.nextActivationFunction.fn(this.outputs);
            } else {
                for (let i = 0; i < this.outputNeuronCount; i++) {
                    this.outputs[i] = this.nextActivationFunction.fn(this.outputs[i]);
                }
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
        learningRate,
        isOutputLayer = false,
        targets = []
    ) {
        var deltas = new Array(this.outputNeuronCount);

        if (isOutputLayer) {
            // Softmax biasanya hanya digunakan pada output layer
            if (this.nextActivationFunction.name === 'softmax') {
                // for (let i = 0; i < this.outputNeuronCount; i++) {
                //     deltas[i] = this.outputs[i] - targets[i];
                // }
                deltas = this.nextActivationFunction.derivative(this.outputs, targets);

            } else {
                for (let i = 0; i < this.outputNeuronCount; i++) {
                    const error = this.outputs[i] - targets[i];
                    deltas[i] = error * this.nextActivationFunction.derivative(this.outputsBeforeActivation[i]);
                }
            }
        } else {
            for (let i = 0; i < this.outputNeuronCount; i++) {
                let error = 0;

                for (let j = 0; j < nextLayerDeltas.length; j++) {
                    error += nextLayerDeltas[j] * nextLayerWeights[j][i];
                }

                deltas[i] = error * this.nextActivationFunction.derivative(this.outputsBeforeActivation[i]);
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