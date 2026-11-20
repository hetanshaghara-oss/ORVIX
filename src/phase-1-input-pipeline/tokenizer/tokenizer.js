function tokenizer(text) {
  if (text.trim() === "") {
    return [];
  }

  return text.trim().split(/\s+/);
}

const tokens = tokenizer("I love ORVIX AI");

// console.log(tokens);

function buildVocablary(tokens) {
  const vocablary = {
    "<BOS>": 0,
    "<EOS>": 1,
    "<PAD>": 2,
    "<UNK>": 3,
  };

  tokens.forEach((token) => {
    if (vocablary[token] == undefined) {
      const tokenId = Object.keys(vocablary).length;

      vocablary[token] = tokenId;
    }
  });

  return vocablary;
}

const vocablary = buildVocablary(tokens);
// console.log(vocablary);

// find the tokenID for each token

function encode(tokens, vocablary) {
  const tokenIds = [];

  tokens.forEach((token) => {
    let id;

    if (vocablary[token] !== undefined) {
      id = vocablary[token];
    } else {
      id = vocablary["<UNK>"];
    }

    tokenIds.push(id);
  });

  return tokenIds;
}

const tokenIDS = encode(tokens, vocablary);
// console.log(tokenIDS);

function decode(tokenids, vocablary) {
  const reverceVocablary = [];

  for (let token in vocablary) {
    const id = vocablary[token];

    reverceVocablary[id] = token;
  }

  const words = [];

  tokenids.forEach((id) => {
    const word = reverceVocablary[id];
    words.push(word);
  });

  return words;
}

const decodeWord = decode(tokenIDS, vocablary);
// console.log(decodeWord);

function padsequence(sequence, maxLength, padId) {
  while (sequence.length < maxLength) {
    sequence.push(padId);
  }
  return sequence;
}

// padding mask

function paddingMask(sequence, padID) {
  const mask = [];

  sequence.forEach((tokenId) => {
    if (tokenId == padID) {
      mask.push(0);
    } else {
      mask.push(1);
    }
  });

  return mask;
}

//add rhe BOS &  EOS

function addSpecialTokens(tokenIDS, vocablary) {
  return [vocablary["<BOS>"], ...tokenIDS, vocablary["<EOS>"]];
}

const finalSeq = addSpecialTokens(tokenIDS, vocablary);

const padding = padsequence(finalSeq, 8, vocablary["<PAD>"]);

const mask = paddingMask(padding, vocablary["<PAD>"]);

// console.log("final seq", finalSeq);
// console.log("padding", padding);
// console.log("mask", mask);

/// embedding

const embeddingMatrix = [
  [0.1, 0.2, 0.3, 0.4], // ID 0
  [0.5, 0.6, 0.7, 0.8], // ID 1
  [0.0, 0.0, 0.0, 0.0], // ID 2
  [0.2, 0.1, 0.4, 0.3], // ID 3
  [0.8, 0.2, 0.5, 0.1], // ID 4
  [0.3, 0.9, 0.2, 0.6], // ID 5
  [0.7, 0.4, 0.8, 0.2], // ID 6
  [0.6, 0.3, 0.5, 0.9],
];

function getEmbedding(tokenIDS, embeddingMatrix) {
  return tokenIDS.map((id) => {
    return embeddingMatrix[id];
  });
}

const embedding = getEmbedding(padding, embeddingMatrix);
// console.log("embedding", embedding);

// postional encoding

//darek one position mate aape value
function positionalEncoding(postion, dModel) {
  const Encoding = [];

  for (let i = 0; i < dModel; i++) {
    const angle = postion / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);

    if (i % 2 == 0) {
      Encoding.push(Math.sin(angle));
    } else {
      Encoding.push(Math.cos(angle));
    }
  }
  return Encoding;
}

// badhi postion ni calculate kari aape
function createPositionalEncoding(sequenceLength, dModel) {
  let positionalMatrix = [];

  for (let pos = 0; pos < sequenceLength; pos++) {
    const encoding = positionalEncoding(pos, dModel);
    positionalMatrix.push(encoding);
  }

  return positionalMatrix;
}

const positionalMatrix = createPositionalEncoding(embedding.length, 4);

// console.log("positionalMatrix", positionalMatrix);

function addpositionalEncoding(embeddings, positionalMatrix) {
  return embeddings.map((vector, pos) => {
    return vector.map((value, dim) => {
      return value + positionalMatrix[pos][dim];
    });
  });
}

const finalInput = addpositionalEncoding(embedding, positionalMatrix);
// console.log(finalInput);

//multiplication

function matrixMultiplication(A, B) {
  const rowsA = A.length;
  const colsA = A[0].length;

  const rowsB = B.length;
  const colsB = B[0].length;

  if (colsA !== rowsB) {
    throw new Error("Matrix multiplication is not possible");
  }

  const result = [];

  for (let i = 0; i < rowsA; i++) {
    const row = [];

    for (let j = 0; j < colsB; j++) {
      let sum = 0;

      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * B[k][j];
      }

      row.push(sum);
    }

    result.push(row);
  }

  return result;
}

//weight matrix
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

const Q = matrixMultiplication(finalInput, WQ);

const K = matrixMultiplication(finalInput, WK);

const V = matrixMultiplication(finalInput, WV);

// console.log("Query:", Q);
// console.log("Key:", K);
// console.log("Value:", V);

// attension score

function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;

  const result = [];

  for (let j = 0; j < cols; j++) {
    const row = [];

    for (let i = 0; i < rows; i++) {
      row.push(matrix[i][j]);
    }

    result.push(row);
  }

  return result;
}

//calculate the attension score

const KT = transpose(K);
const attensionScore = matrixMultiplication(Q, KT);
// console.log("attensionScore:-", attensionScore);

//scale atternsion Score

function scaleAttensionScore(scores, dk) {
  const scale = Math.sqrt(dk);
  return scores.map((row) => {
    return row.map((value) => {
      return value / scale;
    });
  });
}

const scaledScores = scaleAttensionScore(attensionScore, 4);

// console.log(scaledScores);

//causal matrix

function createCausalMask(sequenceLength) {
  const mask = [];

  for (let i = 0; i < sequenceLength; i++) {
    const row = [];
    for (let j = 0; j < sequenceLength; j++) {
      if (j <= i) {
        row.push(0);
      } else {
        row.push(-Infinity);
      }
    }
    mask.push(row);
  }
  return mask;
}

const sequenceLength = finalInput.length;
const causalMask = createCausalMask(sequenceLength);

//addition of the scaled attension score and the causalMask score

function applyCausalMask(attentionScores, causalMask) {
  const result = [];

  for (let i = 0; i < attentionScores.length; i++) {
    const row = [];

    for (let j = 0; j < attentionScores[i].length; j++) {
      row.push(attentionScores[i][j] + causalMask[i][j]);
    }

    result.push(row);
  }

  return result;
}

const markedScore = applyCausalMask(scaledScores, causalMask);
// console.log("Markes Score is the :-", markedScore);

//apply padding mask

function applupaddingMask(attensionScores, paddingMask) {
  return attensionScores.map((row) => {
    return row.map((score, j) => {
      return paddingMask[j] === 0 ? -Infinity : score;
    });
  });
}

const finalMarkedScore = applupaddingMask(markedScore, mask);

//softmax

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const expScore = scores.map((score) => {
    return Math.exp(score - maxScore);
  });

  const sum = expScore.reduce((acc, curr) => {
    return acc + curr;
  }, 0);

  const probabolity = expScore.map((score) => {
    return score / sum;
  });

  return probabolity;
}

function softMatrix(matrix) {
  return matrix.map((row) => {
    return softmax(row);
  });
}

const attensionWeight = softMatrix(finalMarkedScore);

// console.log("Attension Weight :-", attensionWeight);

//attension output

const attensionOutput = matrixMultiplication(attensionWeight, V);

// console.log(attensionOutput);

module.exports = {
  finalInput,
};
