import { argsort } from "./helpers";
import { Tensor } from "./tensor";

export type Op = {
  inputs: Tensor[];
  backward(resultGrad: Tensor): Tensor[];
};

/* Binary ops */

export class AddOp implements Op {
  inputs: [Tensor, Tensor];

  public constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad, resultGrad];
  }
}

export class MulOp implements Op {
  inputs: [Tensor, Tensor];

  public constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  public backward(resultGrad: Tensor) {
    const [x, y] = this.inputs;

    return [y.mul(resultGrad, false), x.mul(resultGrad, false)];
  }
}

/* Movement ops */

export class ReshapeOp implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [resultGrad.reshape(x.shape, false)];
  }
}

export class PermuteOp implements Op {
  inputs: [Tensor];
  order: number[];

  public constructor(x: Tensor, order: number[]) {
    this.inputs = [x];
    this.order = order;
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad.permute(argsort(this.order), false)];
  }
}

export class ExpandOp implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    // TODO $multi-axis-sum$
    const sumAxis = x.shape.findIndex((value, i) => value === 1 && resultGrad.shape[i] !== 1);

    return [resultGrad.sum(sumAxis, false, true)];
  }
}

/* Reduce ops */

export class SumOp implements Op {
  inputs: [Tensor];
  axis: number;
  keepDim: boolean;

  public constructor(x: Tensor, axis: number, keepDim: boolean) {
    this.inputs = [x];
    this.axis = axis;
    this.keepDim = keepDim;
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    const shapeToExpandFrom = [...resultGrad.shape];

    if (!this.keepDim) {
      // with keepDim=false, sum removes reduced dimension from shape completely,
      // so we need to add it back before expanding
      shapeToExpandFrom.splice(this.axis, 0, 1);
    }

    return [resultGrad.reshape(shapeToExpandFrom, false).expand(x.shape, false)];
  }
}
