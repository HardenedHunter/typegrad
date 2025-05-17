import { argsort } from "./helpers";
import { Tensor } from "./tensor";

export type Op = {
  inputs: Tensor[];
  backward(resultGrad: Tensor): Tensor[];
};

/* Binary ops */

export class Add implements Op {
  inputs: [Tensor, Tensor];

  public constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad, resultGrad];
  }
}

export class Sub implements Op {
  inputs: [Tensor, Tensor];

  public constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad, resultGrad.neg(false)];
  }
}

export class Mul implements Op {
  inputs: [Tensor, Tensor];

  public constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  public backward(resultGrad: Tensor) {
    const [x, y] = this.inputs;

    return [y.mul(resultGrad, false), x.mul(resultGrad, false)];
  }
}

/* Unary ops */

export class Sign implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad.zerosLike()];
  }
}

export class ReLU implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [x.sign(false).relu(false).mul(resultGrad, false)];
  }
}

export class Log implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [x.reciprocal(false).mul(resultGrad, false)];
  }
}

export class Exp implements Op {
  inputs: [Tensor];
  result: Tensor;

  public constructor(x: Tensor, result: Tensor) {
    this.inputs = [x];
    this.result = result;
  }

  public backward(resultGrad: Tensor) {
    return [this.result.mul(resultGrad, false)];
  }
}

export class Reciprocal implements Op {
  inputs: [Tensor];
  result: Tensor;

  public constructor(x: Tensor, result: Tensor) {
    this.inputs = [x];
    this.result = result;
  }

  public backward(resultGrad: Tensor) {
    return [resultGrad.neg(false).mul(this.result, false).mul(this.result, false)];
  }
}

export class Neg implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    return [Tensor.full(resultGrad.shape, -1).mul(resultGrad, false)];
  }
}

/* Movement ops */

export class Reshape implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [resultGrad.reshape(x.shape, false)];
  }
}

export class Permute implements Op {
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

export class Expand implements Op {
  inputs: [Tensor];

  public constructor(x: Tensor) {
    this.inputs = [x];
  }

  public backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    const countOfExpandingDims = x.shape.reduce((acc, s, i) => (s === 1 && resultGrad.shape[i] !== 1 ? acc + 1 : acc), 0);

    // TODO $multi-axis-sum$
    if (countOfExpandingDims > 1) {
      throw new Error(
        `[Expand.backward] can't do backward when expanding >1 dim at a time as sum currently doesn't support multiple axes, expanding ${x.shape} -> ${resultGrad.shape}`
      );
    }

    const sumAxis = x.shape.findIndex((value, i) => value === 1 && resultGrad.shape[i] !== 1);

    return [resultGrad.sum(sumAxis, false, true)];
  }
}

/* Reduce ops */

export class Sum implements Op {
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
