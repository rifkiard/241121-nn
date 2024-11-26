const fs = require('fs');
const { createCanvas } = require('canvas');

function normalizeData(data, min = null, max = null) {

    if (min === null) {
        max = Math.max(...data);
        min = Math.min(...data);
    }
    return data.map(x => (x - min) / (max - min));
}

function normalizeDataSingle(value, min, max) {
    return (value - min) / (max - min);
}

function renderImage(data, width, height, outputPath) {
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < data.length; i++) {
        const value = data[i]; // Grayscale value (0-255)
        const pixelIndex = i * 4; // RGBA has 4 components per pixel

        imageData.data[pixelIndex] = value;     // Red
        imageData.data[pixelIndex + 1] = value; // Green
        imageData.data[pixelIndex + 2] = value; // Blue
        imageData.data[pixelIndex + 3] = 255;   // Alpha
    }

    ctx.putImageData(imageData, 0, 0);

    const out = fs.createWriteStream(outputPath);
    const stream = canvas.createPNGStream();
    stream.pipe(out);
    out.on('finish', () => console.log(`Image saved to ${outputPath}`));
}

function createOneHotVector(label, vectorLength) {
    const vector = new Array(vectorLength).fill(0);
    vector[parseInt(label)] = 1;
    return vector;
}

module.exports = {
    normalizeData,
    normalizeDataSingle,
    renderImage,
    createOneHotVector
}