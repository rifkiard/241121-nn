const fs = require("fs");
const { renderImage, normalizeData, createOneHotVector } = require("../src/Utils");
const NeuralNetwork = require('../src/NeuralNetwork');

const data = fs.readFileSync(__dirname + '/mnist_test.csv', "utf-8");
const rows = data.split("\n").filter(row => row.trim() !== '');

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}


var mnist = rows.map((row, index) => {
    const columns = row.split(",");
    const pixels = normalizeData(columns.slice(1).map(Number));

    return {
        label: columns[0],
        oneHotLabel: createOneHotVector(columns[0], 11),
        pixels: pixels
    };
});

shuffleArray(mnist);

const nn = NeuralNetwork.load(__dirname + '/model.json');

let correct = 0;
let total = 0;

mnist.forEach((data, index) => {
    const output = nn.feedForward(data.pixels);
    const prediction = output.indexOf(Math.max(...output));
    const target = data.oneHotLabel.indexOf(1);

    // if (prediction !== target) {
    //     console.warn(`Prediction: ${prediction}, Target: ${target} (${prediction === target ? 'Pass' : 'Failed'})`);
    // } else {
    //     console.log(`Prediction: ${prediction}, Target: ${target} (${prediction === target ? 'Pass' : 'Failed'})`);
    // }
    if (prediction === target) {
        correct++;
    }

    total++;
});

console.log(`Accuracy: ${(correct / total) * 100}%`);