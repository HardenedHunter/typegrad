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
};

const crossEntropyLoss = (label: Tensor, logOfPredicted: Tensor) => label.mul(logOfPredicted).sum(0).neg();

for (let i = 0; i < trainImages.length; i += 784) {
  const x = new Tensor(new Float32Array([...trainImages.subarray(i, i + 784)].map((value) => value / 255)), new View([784]));
  xTrain.push(x);
}

for (let i = 0; i < trainLabels.length; i++) {
  yTrain.push(Tensor.from(oneHot(trainLabels[i], 10)));
}

const lr = 0.1;

class Model {
  w1 = Tensor.zeros([784, 10]);
  b1 = Tensor.zeros([10]);

  forward(x: Tensor) {
    return x.dot(this.w1).add(this.b1).logSoftmax();
  }

  step() {
    this.w1 = this.w1.sub(this.w1.grad!.mul(lr, false), false);
    this.b1 = this.b1.sub(this.b1.grad!.mul(lr, false), false);
  }

  zeroGrad() {
    this.w1.grad = undefined;
    this.b1.grad = undefined;
  }
}

const model = new Model();

const trainSize = 2000;
const testSize = 200;

const train = (epoch: number) => {
  let correct = 0;

  for (let i = 0; i < trainSize; i++) {
    model.zeroGrad();
    
    const X = xTrain[i],
      y = yTrain[i];
    const pred = model.forward(X);
    const loss = crossEntropyLoss(y, pred);
    
    loss.backward();
    model.step();

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

for (let i = 0; i < 10; i++) {
  train(i + 1);
  test(i + 1);
}
