import { duration } from "./debug";
import { flatten, NDArray, prod } from "./helpers";
import { View } from "./view";
import { Op, AddOp, ExpandOp, MulOp, PermuteOp, ReshapeOp, SumOp } from "./ops";

export class Tensor {
  public readonly data: Float32Array;
  public readonly view: View;
  public grad?: Tensor;
  public op?: Op;

  private constructor(data: Float32Array, view: View, op?: Op, grad?: Tensor) {
    this.data = data;
    this.view = view;
    this.op = op;
    this.grad = grad;
  }

  public static from(ndArray: NDArray) {
    return new Tensor(new Float32Array(flatten(ndArray)), View.fromNDArray(ndArray));
  }

  @duration()
  public static rand(...shape: number[]) {
    const size = prod(shape);
    const data = new Float32Array(size);

    for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;

    return new Tensor(data, new View(shape));
  }

  @duration()
  public static full(shape: number[], fillValue: number) {
    return new Tensor(new Float32Array(prod(shape)).fill(fillValue), new View(shape));
  }

  public static ones(shape: number[]) {
    return Tensor.full(shape, 1);
  }

  @duration()
  public static zeros(shape: number[]) {
    return new Tensor(new Float32Array(prod(shape)), new View(shape));
  }

  public onesLike() {
    return Tensor.ones(this.shape);
  }

  public zerosLike() {
    return Tensor.zeros(this.shape);
  }

  public get ndim() {
    return this.shape.length;
  }

  public get shape() {
    return this.view.shape;
  }

  public indices() {
    return this.view.indices();
  }

  public get(indices: number[]) {
    if (indices.length !== this.ndim) throw new Error(`[Tensor.get] sub-tensor indexing is not supported for now`);

    const index = indices.reduce((acc, index, i) => acc + index * this.view.strides[i], 0);

    return this.data[index];
  }

  public set(indices: number[], value: number) {
    if (indices.length !== this.ndim) throw new Error(`[Tensor.set] sub-tensor assignment is not supported for now`);

    const index = indices.reduce((acc, index, i) => acc + index * this.view.strides[i], 0);

    this.data[index] = value;
  }

  @duration()
  public permute(order: number[], requiresGrad = true) {
    const view = this.view.permute(order);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (requiresGrad) {
      result.op = new PermuteOp(this, order);
    }

    return result;
  }

  @duration()
  public reshape(newShape: number[], requiresGrad = true) {
    const view = this.view.reshape(newShape);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (requiresGrad) {
      result.op = new ReshapeOp(this);
    }

    return result;
  }

  @duration()
  public expand(newShape: number[], requiresGrad = true) {
    const view = this.view.expand(newShape);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (requiresGrad) {
      result.op = new ExpandOp(this);
    }

    return result;
  }

  @duration()
  public transpose(dim0 = 1, dim1 = 0, requiresGrad = true) {
    dim0 = dim0 < 0 ? this.ndim + dim0 : dim0;
    dim1 = dim1 < 0 ? this.ndim + dim1 : dim1;

    if (dim0 > this.ndim - 1) throw new Error(`[Tensor.transpose] dim0=${dim0} is out of bounds for tensor with ndim=${this.ndim}`);
    if (dim1 > this.ndim - 1) throw new Error(`[Tensor.transpose] dim1=${dim1} is out of bounds for tensor with ndim=${this.ndim}`);

    const order = this.shape.map((_, i) => i);
    order[dim0] = dim1;
    order[dim1] = dim0;

    return this.permute(order, requiresGrad);
  }

  public get T(): Tensor {
    return this.transpose();
  }

  private static broadcasted(a: Tensor, b: Tensor) {
    const maxDim = Math.max(a.shape.length, b.shape.length);
    const aShapeAligned = [...Array(maxDim - a.shape.length).fill(1), ...a.shape];
    const bShapeAligned = [...Array(maxDim - b.shape.length).fill(1), ...b.shape];

    const canBroadcast = aShapeAligned.every((ai, i) => ai === bShapeAligned[i] || ai === 1 || bShapeAligned[i] === 1);

    if (!canBroadcast) throw new Error(`[Tensor.broadcasted] shape values mismatch, can't broadcast ${a.shape} and ${b.shape}`);

    const shape = aShapeAligned.map((ai, i) => Math.max(ai, bShapeAligned[i]));

    return [a.expand(shape), b.expand(shape)];
  }

  @duration()
  public dot(tensor: Tensor) {
    let a: Tensor = this,
      b = tensor;

    const aAxis = -1;
    const bAxis = -Math.min(b.ndim, 2);

    if (a.shape.at(aAxis) !== b.shape.at(bAxis)) throw new Error(`[Tensor.dot] invalid shapes: ${a.shape} and ${b.shape}`);

    const filler = a.ndim < 2 || b.ndim < 2 ? [] : [1];

    a = a.reshape([...a.shape.slice(0, aAxis), ...filler, ...a.shape.slice(aAxis)]);
    b = b.reshape([...b.shape.slice(0, bAxis), ...filler, ...b.shape.slice(bAxis)]).transpose(-1, bAxis);

    return a.mul(b).sum(-1);
  }

  @duration()
  public mul(tensor: Tensor, requiresGrad = true) {
    const [a, b] = Tensor.broadcasted(this, tensor);

    const result = a.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, a.get(indices) * b.get(indices));
    }

    if (requiresGrad) {
      result.op = new MulOp(a, b);
    }

    return result;
  }

  @duration()
  public add(tensor: Tensor, requiresGrad = true) {
    const [a, b] = Tensor.broadcasted(this, tensor);

    const result = a.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, a.get(indices) + b.get(indices));
    }

    if (requiresGrad) {
      result.op = new AddOp(a, b);
    }

    return result;
  }

  @duration()
  public sum(axis: number, requiresGrad = true, keepDim = false) {
    // TODO $multi-axis-sum$
    axis = axis < 0 ? this.ndim + axis : axis;

    if (axis > this.ndim - 1) throw new Error(`[Tensor.sum] axis=${axis} is out of range for tensor with ndim=${this.ndim}`);

    const shape = [...this.shape];
    const sumDimSize = shape[axis];

    if (keepDim) {
      shape.splice(axis, 1, 1);
    } else {
      shape.splice(axis, 1);
    }

    const result = Tensor.zeros(shape);

    for (let indices of result.indices()) {
      let value = 0;

      for (let i = 0; i < sumDimSize; i++) {
        value += this.get([...indices.slice(0, axis), i, ...indices.slice(axis + 1)]);
      }

      result.set(indices, value);
    }

    if (requiresGrad) {
      result.op = new SumOp(this, axis, keepDim);
    }

    return result;
  }

  public backward() {
    if (!this.grad) this.grad = this.onesLike();

    if (!this.op) return;

    const grads = this.op.backward(this.grad);

    this.op.inputs.forEach((input, i) => {
      input.grad = input.grad ? input.grad.add(grads[i], false) : grads[i];
      input.backward();
    });
  }

  public render() {
    const shape = this.shape;
    const strides = this.view.strides;

    const renderChunk = (offset: number, axis: number, pos: number): string => {
      const lastAxis = axis === this.ndim - 1;

      const items = Array.from({ length: shape[axis] }, (_, i) => {
        if (lastAxis) return this.data[offset + i * strides[axis]];

        return renderChunk(offset + i * strides[axis], axis + 1, i);
      });

      const joined = lastAxis ? items.join(", ") : items.join(",\n");
      const indent = pos > 0 ? " ".repeat(axis) : "";

      return `${indent}[${joined}]`;
    };

    return renderChunk(0, 0, 0);
  }
}
