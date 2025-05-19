import fs from "node:fs";
import { View } from "./view";
import { Tensor } from "./tensor";
import { random } from "./random";

random.seed(0);

const trainImages = fs.readFileSync("datasets/mnist/train-images-idx3-ubyte").subarray(0x10);
const trainLabels = fs.readFileSync("datasets/mnist/train-labels-idx1-ubyte").subarray(8);

const xTrain: Tensor[] = [];
const yTrain: Tensor[] = [];

const oneHot = (label: number, classes: number) => {
  const result = Array(classes).fill(0);

  result[label] = 1;

  return result;
};

for (let i = 0; i < trainImages.length; i += 784) {
  const x = new Tensor(new Float32Array([...trainImages.subarray(i, i + 784)].map((value) => value / 255)), new View([784]));
  xTrain.push(x);
}

for (let i = 0; i < trainLabels.length; i++) {
  yTrain.push(Tensor.from(oneHot(trainLabels[i], 10)));
}

class Model {
  w1: Tensor;
  w2: Tensor;

  constructor() {
    this.w1 = Tensor.kaimingNormal([784, 32], undefined, true);
    this.w2 = Tensor.kaimingNormal([32, 10], undefined, true);
  }

  forward(x: Tensor) {
    return x.dot(this.w1).relu().dot(this.w2);
  }

  step(learningRate: number) {
    this.w1 = this.w1.detach().sub(this.w1.grad!.mul(learningRate));
    this.w2 = this.w2.detach().sub(this.w2.grad!.mul(learningRate));
    this.w1.requiresGrad = true;
    this.w2.requiresGrad = true;
  }

  zeroGrad() {
    this.w1.grad = undefined;
    this.w2.grad = undefined;
  }
}

const model = new Model();

const trainSize = 2000;
const testSize = 200;
const batchSize = 32;
const learningRate = 0.1;
const epochs = 10;

const train = (epoch: number) => {
  let correct = 0;

  model.zeroGrad();

  for (let i = 0; i < trainSize; i++) {
    const X = xTrain[i],
      y = yTrain[i];
    const pred = model.forward(X);
    const loss = pred.categoricalCrossEntropy(y);

    loss.backward();

    const endOfBatch = (i + 1) % batchSize === 0 || i === trainSize - 1;

    if (endOfBatch) {
      model.step(learningRate / batchSize);
      model.zeroGrad();
    }

    correct += pred.data.indexOf(Math.max(...pred.data)) === y.data.indexOf(1) ? 1 : 0;
  }

  console.log(`Train Epoch ${epoch} ${((correct / trainSize) * 100).toFixed(2)}%`);
};

const test = (epoch: number) => {
  let correct = 0;

  for (let i = trainSize; i < trainSize + testSize; i++) {
    const X = xTrain[i],
      y = yTrain[i];
    const pred = model.forward(X);

    correct += pred.data.indexOf(Math.max(...pred.data)) === y.data.indexOf(1) ? 1 : 0;
  }

  console.log(`Test Epoch ${epoch} ${((correct / testSize) * 100).toFixed(2)}%`);
};

for (let i = 0; i < epochs; i++) {
  train(i + 1);
  test(i + 1);
}
