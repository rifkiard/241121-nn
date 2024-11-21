const NeuralNetwork = require('./src/NeuralNetwork');
const { relu } = require('./src/ActivationFunctions');


const network = new NeuralNetwork(
    [2, 3, 5, 2],
    [
        relu,
        relu,
        relu,
    ]
);

console.log(network.feedForward([1, 2]));