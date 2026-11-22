const {
  transformerBlockOutput,
} = require("../phase-3-transformer-block/transformer-block.js");

const vocabularys = ["I", "love", "programming", "AI", "Python", "cricket"];

const d_model = transformerBlockOutput[0].length;
const vocablarySize = vocabularys.length;

function initializeWeights(rows, cols) {
  const weight = [];

  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random() * 0.1);
    }
    weight.push(row);
  }

  return weight;
}

function initializeBias(size) {
  return Array(size).fill(0);
}

const W = initializeWeights(d_model, vocablarySize);

const b = initializeBias(vocablarySize);

function matrixMultiplication(A, B) {
  const result = [];

  for (let i = 0; i < A.length; i++) {
    const row = [];

    for (let j = 0; j < B[0].length; j++) {
      let sum = 0;

      for (let k = 0; k < B.length; k++) {
        sum += A[i][k] * B[k][j];
      }

      row.push(sum);
    }

    result.push(row);
  }

  return result;
}

function addBias(matrix, bias) {
  return matrix.map((row) => row.map((value, index) => value + bias[index]));
}

const multiplicationOutput = matrixMultiplication(transformerBlockOutput, W);

const Logits = addBias(multiplicationOutput, b);

const lastTokenLogits = Logits[Logits.length - 1];

//apply the softmax on the last row

function softmax(matrix) {
  const maxLogit = Math.max(...matrix);
  const expValue = matrix.map((value) => {
    return Math.exp(value - maxLogit);
  });

  const sum = expValue.reduce((acc, curr) => {
    return acc + curr;
  }, 0);

  const probabolity = expValue.map((value) => {
    return value / sum;
  });

  return probabolity;
}

const probabilites = softmax(lastTokenLogits);

// greedy section means select the higher probbility token

function greedySection(probabilites, vocabularys) {
  let maxIdx = 0;

  for (let i = 0; i < probabilites.length; i++) {
    if (probabilites[i] > probabilites[maxIdx]) {
      maxIdx = i;
    }
  }

  return {
    token: vocabularys[maxIdx],
    tokenId: maxIdx,
    probabolity: probabilites[maxIdx],
  };
}

const prediction = greedySection(probabilites,vocabularys);
