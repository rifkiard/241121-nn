const fs = require("fs");
const { renderImage, normalizeData, createOneHotVector } = require("../src/Utils");
const NeuralNetwork = require('../src/NeuralNetwork');
const { reLU, softmax } = require("../src/ActivationFunctions");

const data = fs.readFileSync(__dirname + '/mnist_train.csv', "utf-8");
const rows = data.split("\n").filter(row => row.trim() !== '');

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]]; // Swap elements
    }
}

function validateInputData(pixels) {
    // Comprehensive input validation
    if (!Array.isArray(pixels)) {
        console.error('Input is not an array');
        return false;
    }

    // Check for NaN or Infinity
    const hasInvalidNumbers = pixels.some(pixel =>
        pixel === null ||
        pixel === undefined ||
        isNaN(pixel) ||
        !isFinite(pixel)
    );

    if (hasInvalidNumbers) {
        console.error('Invalid numbers detected in input');
        return false;
    }

    // Check value range
    const invalidRange = pixels.some(pixel =>
        typeof pixel !== 'number' ||
        pixel < -1 ||
        pixel > 1
    );

    if (invalidRange) {
        console.error('Input values out of expected range');
        console.log('Sample values:', pixels.slice(0, 10)); // Log first 10 values
        return false;
    }

    return true;
}

var mnist = rows.map((row, index) => {
    try {
        const columns = row.split(",");
        const pixels = normalizeData(columns.slice(1).map(Number));

        // Additional logging for problematic inputs
        if (!validateInputData(pixels)) {
            console.error(`Invalid input at row ${index}`);
            console.log('Raw row:', row);
            console.log('Parsed columns:', columns);
        }

        return {
            label: columns[0],
            oneHotLabel: createOneHotVector(columns[0], 11),
            pixels: pixels
        };
    } catch (error) {
        console.error(`Error processing row ${index}:`, error);
        return null;
    }
}).filter(Boolean);

shuffleArray(mnist);

// const nn = new NeuralNetwork([
//     28 * 28,
//     128,
//     31,
//     11
// ], [
//     reLU,
//     reLU,
//     softmax
// ]);

const nn = NeuralNetwork.load(__dirname + '/model.json');

var test = nn.feedForward(mnist[0].pixels);
console.log("Label: ", mnist[0].label);

const indexOfMax = test.reduce((maxIndex, currentValue, currentIndex, array) =>
    currentValue > array[maxIndex] ? currentIndex : maxIndex, 0);

console.log("test result: ", indexOfMax);
renderImage(mnist[0].pixels.map(x => x * 255), 28, 28, __dirname + '/test.png');

const learningRate = 0.0001;
const epochs = 100_000;


function trainNetwork() {
    const startTime = Date.now();

    console.log("training begin.");


    for (let i = 0; i < epochs; i++) {
        let totalError = 0;
        let processedSamples = 0;

        for (let j = 0; j < mnist.length; j++) {
            const error = nn.backPropagation(
                mnist[j].pixels,
                mnist[j].oneHotLabel,
                learningRate
            );

            totalError += error;
            processedSamples++;

        }

        if (isNaN(totalError)) {
            console.error(`NaN total error detected at epoch ${i}`);
            throw new Error('Training stopped due to NaN total error');
        }

        let elapsedTime = (Date.now() - startTime) / 1000;

        console.log(`Epoch ${i + 1}, Mean Error: ${totalError / mnist.length}, Accuracy: ${((1 - totalError / mnist.length) * 100).toFixed(4)}%, Elapsed Time: ${elapsedTime} seconds`);

        if (totalError / processedSamples < 1e-5) {
            console.log("Convergence reached, stopping training.");
            break;
        }

        nn.save(__dirname + '/model.json');
    }

    console.log("Training completed.");
}


trainNetwork();

nn.saveModel(__dirname + '/model.json');
