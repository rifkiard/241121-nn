
const { sigmoid } = require('../src/ActivationFunctions');
const NeuralNetwork = require('../src/NeuralNetwork');
const dataset = require('./dataset.json');

function normalizeData(data) {
    const max = Math.max(...data);
    const min = Math.min(...data);
    return data.map(x => (x - min) / (max - min));
}

function normalizeSingle(value, min, max) {
    return (value - min) / (max - min);
}

const studyHours = dataset.data.map(data => data.study_hours);
const passes = dataset.data.map(data => data.pass);
const normalizedHours = normalizeData(studyHours);

const learningRate = 0.1;
const epochs = 10_000_000;

var nn = new NeuralNetwork([1, 3, 1], [sigmoid, sigmoid]);

function trainNetwork() {
    for (let i = 0; i < epochs; i++) {
        let totalError = 0;

        for (let j = 0; j < normalizedHours.length; j++) {
            let error = nn.backPropagation(
                [normalizedHours[j]],
                [passes[j]],
                learningRate
            )

            totalError += error;
        }

        if ((i + 1) % 100 === 0) {
            console.log(`Epoch ${i + 1}, Mean Error: ${totalError / normalizedHours.length}`);
        }
    }

    console.log("Training completed.");

}

function testNetwork() {
    console.log("\nTesting the network:");

    for (let i = 0; i < normalizedHours.length; i++) {
        const input = [normalizedHours[i]];
        const prediction = nn.feedForward(input)[0];

        console.log(`
            Study Hours: ${studyHours[i]} 
            Normalized Input: ${input[0].toFixed(4)}
            Actual Pass: ${passes[i]}
            Predicted Pass Probability: ${prediction.toFixed(4)}
        `);
    }
}

// trainNetwork();

// nn.save('exam-pass-predict/model.json');

nn = nn.load('exam-pass-predict/model.json');

testNetwork();

console.log(nn.feedForward([normalizeSingle(1, 0, 8)])[0].toFixed(4));