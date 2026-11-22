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
const attensionScore2 = matrixMultiply(QHeads[1], transpose(KHeads[1])); //8x8

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

//causalMask

function causalMask(matrix) {
  const result = [];

  for (let i = 0; i < matrix.length; i++) {
    let row = [];

    for (let j = 0; j < matrix[i].length; j++) {
      if (j > i) {
        row.push(-Infinity);
      } else {
        row.push(matrix[i][j]);
      }
    }
    result.push(row);
  }
  return result;
}

const markedAttentionScore1 = causalMask(scaleAttensionScore1);
const markedAttentionScore2 = causalMask(scaleAttensionScore2);

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

  return result;
}

const attensionWeight1 = softmax(markedAttentionScore1);
const attensionWeight2 = softmax(markedAttentionScore2);

// attension HeadOutput = attensionweight X v     (8 \times 8) \times (8 \times 2) = 8 \times 2

const headOutput1 = matrixMultiply(attensionWeight1, VHead[0]);
const headOutput2 = matrixMultiply(attensionWeight2, VHead[1]);

//concatinations

function concatenateHead(head1, head2) {
  const result = [];

  for (let i = 0; i < head1.length; i++) {
    const row = head1[i].concat(head2[i]);

    result.push(row);
  }
  return result;
}

const concatinationsOutput = concatenateHead(headOutput1, headOutput2); //8x4

//  output projection
const Wo = [
  [0.2, 0.1, 0.3, 0.4],
  [0.4, 0.3, 0.2, 0.1],
  [0.1, 0.5, 0.4, 0.2],
  [0.3, 0.2, 0.1, 0.5],
];

const finalOutputProjection = matrixMultiply(concatinationsOutput, Wo);

console.log("Final Attention Output:-", finalOutputProjection);

module.exports = {
  finalAttentionOutput: finalOutputProjection,
};

/* 
Input X           8 × 4
      |
   Q, K, V        8 × 4
      |
  Split Heads     8 × 2
      |
  Attention       8 × 8
      |
   Scaling        8 × 8
      |
 Causal Mask      8 × 8
      |
   Softmax        8 × 8
      |
 Attention × V   8 × 2
      |
 Concatenation    8 × 4
      |
 Output Wo       8 × 4
      |
      ▼
 Final Output     8 × 4




*/

module.exports = {};






