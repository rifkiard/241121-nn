const Layer = require('./Layer');
const fs = require('fs');


class NeuralNetwork {
    layers;
    activationFunctions;

    constructor(neuronCounts, activationFunctions = []) {
        this.activationFunctions = activationFunctions;

        this.layers = new Array(neuronCounts.length - 1);

        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i] = new Layer(neuronCounts[i], neuronCounts[i + 1], i == 0 ? null : this.activationFunctions[i - 1], this.activationFunctions[i] || null);
        }
    }

    // inputValues sesuai dengan banyak neuron / features.
    feedForward(inputFeatures) {
        if (!inputFeatures || inputFeatures.length === 0) {
            throw new Error('[NeuralNetwork::feedForward] Invalid input features');
        }

        let output = this.layers[0].feedForward(inputFeatures);

        for (let i = 1; i < this.layers.length; i++) {
            output = this.layers[i].feedForward(output);
        }

        return output;
    }

    backPropagation(inputFeatures, targetValues, learningRate) {
        const output = this.feedForward(inputFeatures);

        let nextLayerDeltas = null;
        let nextLayerWeights = null;

        for (let i = this.layers.length - 1; i >= 0; i--) {
            const isOutputLayer = i === this.layers.length - 1;

            nextLayerDeltas = this.layers[i].backPropagate(
                nextLayerDeltas,
                nextLayerWeights,
                learningRate,
                isOutputLayer,
                targetValues
            );

            nextLayerWeights = this.layers[i].weights;
        }

        // menghitung rata-rata error (squared)

        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += Math.pow(output[i] - targetValues[i], 2);
        }

        error = error / output.length;

        return error;
    }

    save(path) {
        const model = {
            act_func: this.activationFunctions.map(activationFunction => activationFunction.name),
            layers: this.layers.map(function (layer) {
                return {
                    inputNeuronCount: layer.inputNeuronCount,
                    outputNeuronCount: layer.outputNeuronCount,
                    weights: layer.weights,
                    biases: layer.biases
                }
            })
        }

        if (fs.existsSync(path)) {
            fs.unlinkSync(path);
        }

        try {
            fs.writeFileSync(path, JSON.stringify(model));
        } catch (err) {
            console.error(err);
        }
    }

    static load(path) {
        try {
            const model = JSON.parse(fs.readFileSync(path, 'utf8'));

            const activationFunctions = model.act_func.map(func => {
                return require(`./ActivationFunctions`)[func];
            })

            var neurons = [];

            model.layers.forEach((layer, index) => {
                if (index == 0) {
                    neurons.push(layer.inputNeuronCount);
                }

                neurons.push(layer.outputNeuronCount);
            });

            const nn = new NeuralNetwork(neurons, activationFunctions);

            model.layers.forEach((layer, index) => {
                nn.layers[index].weights = layer.weights;
                nn.layers[index].biases = layer.biases;
            })

            return nn;
        } catch (error) {

            console.error('Error loading model:', error);
            return null;
        }
    }

}

module.exports = NeuralNetwork;