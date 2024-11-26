function normalizeData(data) {
    const max = Math.max(...data);
    const min = Math.min(...data);
    return data.map(x => (x - min) / (max - min));
}

function normalizeSingle(value, min, max) {
    return (value - min) / (max - min);
}

module.exports = {
    normalizeData,
    normalizeSingle
}