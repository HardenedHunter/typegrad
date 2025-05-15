import fs from "node:fs";
import { View } from "./view";
import { Tensor } from "./tensor";

const trainImages = fs.readFileSync("datasets/mnist/train-images-idx3-ubyte").subarray(0x10);
const trainLabels = fs.readFileSync("datasets/mnist/train-labels-idx1-ubyte").subarray(8);

const xTrain: Tensor[] = [];
const yTrain: Tensor[] = [];

const oneHot = (label: number, classes: number) => {
  const result = Array(classes).fill(0);

  result[label] = 1;

  return result;
}

for (let i = 0; i < trainImages.length; i += 784) {
  const x = new Tensor(new Float32Array([...trainImages.subarray(i, i + 784)].map(value => value / 255)), new View([784]));
  xTrain.push(x);
}

for (let i = 0; i < trainLabels.length; i++) {
  yTrain.push(Tensor.from(oneHot(trainLabels[i], 10)));
}

let w1 = Tensor.full([784, 10], 1e-2);
let b1 = Tensor.full([10], 1e-2);

const lr = 0.1;
const lr_w1 = Tensor.full(w1.shape, lr);
const lr_b1 = Tensor.full(b1.shape, lr);

const trainSize = 2000;
const testSize = 200;

const train = (epoch: number) => {
  let correct = 0;

  for (let i = 0; i < trainSize; i++) {
    w1.grad = undefined;
    b1.grad = undefined;
    
    const X = xTrain[i], y = yTrain[i];

    const pred = X.dot(w1).add(b1).logSoftmax();

    const loss = y.mul(pred).sum(0).neg();
    
    loss.backward();
    
    w1 = w1.sub(w1.grad!.mul(lr_w1, false), false);
    b1 = b1.sub(b1.grad!.mul(lr_b1, false), false);
    
    correct += pred.data.indexOf(Math.max(...pred.data)) === y.data.indexOf(1) ? 1 : 0;
  }

  console.log(`Train Epoch ${epoch} ${((correct / trainSize) * 100).toFixed(2)}%`)
};

const test = (epoch: number) => {
  let correct = 0;

  for (let i = trainSize; i < trainSize + testSize; i++) {
    const X = xTrain[i], y = yTrain[i];

    const pred = X.dot(w1, false).add(b1, false).logSoftmax(false);
    
    correct += pred.data.indexOf(Math.max(...pred.data)) === y.data.indexOf(1) ? 1 : 0;
  }

  console.log(`Test Epoch ${epoch} ${((correct / testSize) * 100).toFixed(2)}%`)
};

for (let i = 0; i < 10; i++) {
  train(i + 1);
  test(i + 1);
}
