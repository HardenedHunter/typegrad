import { argsort } from "./helpers";
import { Tensor } from "./tensor";

export type Op = {
  inputs: Tensor[];
  backward(resultGrad: Tensor): Tensor[];
};

/* Binary ops */

export class Add implements Op {
  inputs: [Tensor, Tensor];

  constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  backward(resultGrad: Tensor) {
    return [resultGrad, resultGrad];
  }
}

export class Sub implements Op {
  inputs: [Tensor, Tensor];

  constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  backward(resultGrad: Tensor) {
    return [resultGrad, resultGrad.neg()];
  }
}

export class Mul implements Op {
  inputs: [Tensor, Tensor];

  constructor(x: Tensor, y: Tensor) {
    this.inputs = [x, y];
  }

  backward(resultGrad: Tensor) {
    const [x, y] = this.inputs;

    return [y.detach().mul(resultGrad), x.detach().mul(resultGrad)];
  }
}

/* Unary ops */

export class Sign implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    return [resultGrad.zerosLike()];
  }
}

export class ReLU implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [x.detach().sign().relu().mul(resultGrad)];
  }
}

export class Sqrt implements Op {
  inputs: [Tensor];
  result: Tensor;

  constructor(x: Tensor, result: Tensor) {
    this.inputs = [x];
    this.result = result;
  }

  backward(resultGrad: Tensor) {
    return [this.result.detach().mul(2).reciprocal().mul(resultGrad)];
  }
}

export class Sin implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [x.detach().cos().mul(resultGrad)];
  }
}

export class Log implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [x.detach().reciprocal().mul(resultGrad)];
  }
}

export class Exp implements Op {
  inputs: [Tensor];
  result: Tensor;

  constructor(x: Tensor, result: Tensor) {
    this.inputs = [x];
    this.result = result;
  }

  backward(resultGrad: Tensor) {
    return [this.result.detach().mul(resultGrad)];
  }
}

export class Reciprocal implements Op {
  inputs: [Tensor];
  result: Tensor;

  constructor(x: Tensor, result: Tensor) {
    this.inputs = [x];
    this.result = result;
  }

  backward(resultGrad: Tensor) {
    return [resultGrad.neg().mul(this.result.detach()).mul(this.result.detach())];
  }
}

export class Neg implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    return [Tensor.full(resultGrad.shape, -1).mul(resultGrad)];
  }
}

/* Movement ops */

export class Reshape implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    return [resultGrad.reshape(x.shape)];
  }
}

export class Permute implements Op {
  inputs: [Tensor];
  order: number[];

  constructor(x: Tensor, order: number[]) {
    this.inputs = [x];
    this.order = order;
  }

  backward(resultGrad: Tensor) {
    return [resultGrad.permute(argsort(this.order))];
  }
}

export class Expand implements Op {
  inputs: [Tensor];

  constructor(x: Tensor) {
    this.inputs = [x];
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    const countOfExpandingDims = x.shape.reduce((acc, s, i) => (s === 1 && resultGrad.shape[i] !== 1 ? acc + 1 : acc), 0);

    // TODO $multi-axis-sum$
    if (countOfExpandingDims > 1) {
      throw new Error(
        `[Expand.backward] can't do backward when expanding >1 dim at a time as sum currently doesn't support multiple axes, expanding ${x.shape} -> ${resultGrad.shape}`
      );
    }

    const sumAxis = x.shape.findIndex((value, i) => value === 1 && resultGrad.shape[i] !== 1);

    return [resultGrad.sum(sumAxis, true)];
  }
}

/* Reduce ops */

export class Sum implements Op {
  inputs: [Tensor];
  axis: number;
  keepDim: boolean;

  constructor(x: Tensor, axis: number, keepDim: boolean) {
    this.inputs = [x];
    this.axis = axis;
    this.keepDim = keepDim;
  }

  backward(resultGrad: Tensor) {
    const [x] = this.inputs;

    const shapeToExpandFrom = [...resultGrad.shape];

    if (!this.keepDim) {
      // with keepDim=false, sum removes reduced dimension from shape completely,
      // so we need to add it back before expanding
      shapeToExpandFrom.splice(this.axis, 0, 1);
    }

    return [resultGrad.reshape(shapeToExpandFrom).expand(x.shape)];
  }
}
