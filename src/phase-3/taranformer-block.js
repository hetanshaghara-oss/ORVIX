// PHASE 1: Import Input Matrix

const {
  finalInput: X,
} = require("../phase-1-input-pipeline/tokenizer/tokenizer.js");

const gamma = [2, 2, 2, 2];
const beta = [1, 1, 1, 1];

function layerNormalization(X, gamma, beta, epsilon = 1e-5) {
  const means = X.map((row) => {
    const sum = row.reduce((total, value) => {
      return total + value;
    }, 0);

    return sum / row.length;
  });

  // console.log(means);

  const variance = X.map((row, index) => {
    const mean = means[index];

    const squaredDifferences = row.map((value) => {
      return (value - mean) ** 2;
    });

    const sum = squaredDifferences.reduce((total, value) => {
      return total + value;
    }, 0);

    return sum / row.length;
  });

  const standardDeviations = variance.map((value) => {
    return Math.sqrt(value + epsilon);
  });

  const normalizedMatrix = X.map((row, index) => {
    const mean = means[index];
    const standardDeviation = standardDeviations[index];

    return row.map((value) => {
      return (value - mean) / standardDeviation;
    });
  });

  // Step 5: Apply Gamma and Beta

  const finalOutput = normalizedMatrix.map((row) => {
    return row.map((value, index) => {
      return gamma[index] * value + beta[index];
    });
  });

  console.log("Final LayerNorm Output:", finalOutput);

  return finalOutput;
}

// Execute Layer Normalization

const normalizedInput = layerNormalization(X, gamma, beta);

console.log("Normalized Input:", normalizedInput);

// ========================================
// MULTI-HEAD ATTENTION
// ========================================

function multiHeadAttention(input) {
  const numHeads = 2;
  const dModel = input[0].length;
  const headDim = dModel / numHeads;

  if (dModel % numHeads !== 0) {
    throw new Error("dModel must be divisible by numHeads");
  }

  // STEP 1: WEIGHT MATRICES

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

  const Wo = [
    [0.2, 0.1, 0.3, 0.4],
    [0.4, 0.3, 0.2, 0.1],
    [0.1, 0.5, 0.4, 0.2],
    [0.3, 0.2, 0.1, 0.5],
  ];

  // STEP 2: MATRIX MULTIPLICATION

  function matrixMultiply(A, B) {
    const result = [];

    for (let i = 0; i < A.length; i++) {
      const row = [];

      for (let j = 0; j < B[0].length; j++) {
        let sum = 0;

        for (let k = 0; k < A[0].length; k++) {
          sum += A[i][k] * B[k][j];
        }

        row.push(sum);
      }

      result.push(row);
    }

    return result;
  }

  // STEP 3: Q, K, V

  const Q = matrixMultiply(input, WQ);
  const K = matrixMultiply(input, WK);
  const V = matrixMultiply(input, WV);

  // STEP 4: SPLIT HEADS

  function splitHeads(matrix) {
    const heads = [];

    for (let h = 0; h < numHeads; h++) {
      const head = matrix.map((row) => {
        return row.slice(h * headDim, (h + 1) * headDim);
      });

      heads.push(head);
    }

    return heads;
  }

  const QHeads = splitHeads(Q);
  const KHeads = splitHeads(K);
  const VHeads = splitHeads(V);

  // STEP 5: TRANSPOSE

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

  // STEP 6: SCALED DOT PRODUCT ATTENTION

  function scaleAttentionScore(matrix) {
    const scale = Math.sqrt(headDim);

    return matrix.map((row) => {
      return row.map((value) => value / scale);
    });
  }

  // STEP 7: CAUSAL MASK

  function causalMask(matrix) {
    return matrix.map((row, i) => {
      return row.map((value, j) => {
        if (j > i) {
          return -Infinity;
        }

        return value;
      });
    });
  }

  // STEP 8: SOFTMAX

  function softmax(matrix) {
    return matrix.map((row) => {
      const maxScore = Math.max(...row);

      const expValues = row.map((score) => {
        return Math.exp(score - maxScore);
      });

      const total = expValues.reduce((sum, value) => {
        return sum + value;
      }, 0);

      return expValues.map((value) => {
        return value / total;
      });
    });
  }

  // STEP 9: ATTENTION FOR EACH HEAD

  const headOutputs = [];

  for (let h = 0; h < numHeads; h++) {
    const attentionScore = matrixMultiply(QHeads[h], transpose(KHeads[h]));

    const scaledScore = scaleAttentionScore(attentionScore);

    const maskedScore = causalMask(scaledScore);

    const attentionWeights = softmax(maskedScore);

    const headOutput = matrixMultiply(attentionWeights, VHeads[h]);

    headOutputs.push(headOutput);
  }

  // STEP 10: CONCATENATE HEADS

  const concatenatedOutput = [];

  for (let i = 0; i < input.length; i++) {
    const row = [];

    for (let h = 0; h < numHeads; h++) {
      row.push(...headOutputs[h][i]);
    }

    concatenatedOutput.push(row);
  }

  // STEP 11: OUTPUT PROJECTION

  const finalOutputProjection = matrixMultiply(concatenatedOutput, Wo);

  return finalOutputProjection;
}

const finalAttentionOutput = multiHeadAttention(normalizedInput);

function residualConnection(input, attentionOutput) {
  const result = input.map((row, i) => {
    return row.map((value, j) => {
      return value + attentionOutput[i][j];
    });
  });

  return result;
}

const residualOutput = residualConnection(X, finalAttentionOutput);

console.log("First Residual Output:", residualOutput); // Residual Output (X1)

// apply the  X2

const normalizedOutput = layerNormalization(residualOutput, gamma, beta);

console.log("Second LayerNorm Output:", normalizedOutput);

// FFN

// make the both weight matrix like the newrons

//W1

const W1 = Array.from({ length: 4 }, () => {
  return Array.from({ length: 16 }, () => Math.random() * 0.1);
});

const W2 = Array.from({ length: 16 }, () => {
  return Array.from({ length: 4 }, () => Math.random() * 0.1);
});

//create the bias

const b1 = Array.from({ length: 16 }, () => 0);
const b2 = Array.from({ length: 4 }, () => 0);

function firstLinearLayer(input, weights, bias) {
  const result = [];

  for (let i = 0; i < input.length; i++) {
    const row = [];

    for (let j = 0; j < weights[0].length; j++) {
      let sum = 0;

      for (let k = 0; k < input[i].length; k++) {
        sum += input[i][k] * weights[k][j];
      }

      // Add Bias
      row.push(sum + bias[j]);
    }

    result.push(row);
  }

  return result;
}

const hiddenOutput = firstLinearLayer(normalizedOutput, W1, b1);
// FFN = ReLU(X_2W_1+b_1)W_2+b_2

//ReUL

function ReLU(matrix) {
  return matrix.map((row) => {
    return row.map((value) => {
      return Math.max(0, value);
    });
  });
}

const activetedOutput = ReLU(hiddenOutput);

// STEP 5.6: SECOND LINEAR LAYER

function secondLinearLayer(input, weights, bias) {
  const result = [];

  for (let i = 0; i < input.length; i++) {
    const row = [];

    for (let j = 0; j < weights[0].length; j++) {
      let sum = 0;

      for (let k = 0; k < input[i].length; k++) {
        sum += input[i][k] * weights[k][j];
      }

      // Add Bias
      row.push(sum + bias[j]);
    }

    result.push(row);
  }

  return result;
}

const ffnOutput = secondLinearLayer(activetedOutput, W2, b2);

// Transforemer output = residual of the first resudual and the ffOutput

const transformerOutput = residualConnection(residualOutput, ffnOutput);

function transformerBlock(input) {
  // STEP 1: Layer Normalization
  const normalizedInput = layerNormalization(input, gamma, beta);

  // STEP 2: Multi-Head Attention
  const attentionOutput = multiHeadAttention(normalizedInput);

  // STEP 3: First Residual Connection
  const firstResidual = residualConnection(input, attentionOutput);

  // STEP 4: Second Layer Normalization
  const normalizedOutput = layerNormalization(firstResidual, gamma, beta);

  // STEP 5: Feed Forward Network
  const hidden = firstLinearLayer(normalizedOutput, W1, b1);

  const activated = ReLU(hidden);

  const ffnOutput = secondLinearLayer(activated, W2, b2);

  // STEP 6: Second Residual Connection
  const finalOutput = residualConnection(firstResidual, ffnOutput);

  return finalOutput;
}

const output = transformerBlock(X);

module.exports = {
  output,
};
