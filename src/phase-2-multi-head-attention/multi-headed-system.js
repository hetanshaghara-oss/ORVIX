const {
  finalInput: X,
} = require("../phase-1-input-pipeline/tokenizer/tokenizer.js");

const numHeads = 2;
const dModel = 4;
const headDim = dModel / numHeads;

const WQ = [
  [0.1, 0.2, 0.3, 0.4],
  [0.5, 0.6, 0.7, 0.8],
  [0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9],
];

const WK = [
  [0.2, 0.1, 0.3, 0.4],
  [0.4, 0.5, 0.6, 0.7],
  [0.1, 0.3, 0.5, 0.7],
  [0.8, 0.6, 0.4, 0.2],
];

const WV = [
  [0.3, 0.2, 0.1, 0.4],
  [0.5, 0.4, 0.3, 0.2],
  [0.7, 0.6, 0.5, 0.4],
  [0.1, 0.2, 0.3, 0.5],
];

function matrixMultiply(A, B) {
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

const Q = matrixMultiply(X, WQ);
const K = matrixMultiply(X, WK);
const V = matrixMultiply(X, WV);

// console.log("Q Matrix:", Q);
// console.log("K Matrix:", K);
// console.log("V Matrix:", V);

function splitHeads(matrix) {
  const heads = [];

  for (let h = 0; h < numHeads; h++) {
    const head = matrix.map((row) => row.slice(h * headDim, (h + 1) * headDim));

    heads.push(head);
  }

  return heads;
}

const QHeads = splitHeads(Q);
const KHeads = splitHeads(K);
const VHead = splitHeads(V);

// Transpose Matrix

function transpose(matrix) {
  const result = [];

  for (let j = 0; j < matrix[0].length; j++) {
    const row = [];

    for (let i = 0; i < matrix.length; i++) {
      row.push(matrix[i][j]);
    }

    result.push(row);
  }

  return result;
}

const attensionScore1 = matrixMultiply(QHeads[0], transpose(KHeads[0])); //8x8
const attensionScore2 = matrixMultiply(QHeads[1], transpose[KHeads[1]]); //8x8

function scaleAttensionScore(matrix, headDim) {
  const scale = Math.sqrt(headDim);

  const result = [];

  for (let i = 0; i < matrix.length; i++) {
    const row = [];

    for (let j = 0; j < matrix[i].length; j++) {
      row.push(matrix[i][j] / scale);
    }
    result.push(row);
  }

  return result;
}

const scaleAttensionScore1 = scaleAttensionScore(attensionScore1, headDim);
const scaleAttensionScore2 = scaleAttensionScore(attensionScore2, headDim);

function softmax(matrix) {
  const result = [];
  for (let i = 0; i < matrix.length; i++) {
    const expValue = matrix[i].map((score) => {
      return Math.exp(score);
    });

    const total = expValue.reduce((sum, curr) => {
      return sum + curr;
    }, 0);

    const row = expValue.map((value) => {
      return value / total;
    });

    result.push(row);
  }

  return row;
}



const attensionWeight1 = softmax(scaleAttensionScore1);
const attensionWeight2 = softmax(scaleAttensionScore2);

