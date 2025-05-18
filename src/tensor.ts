import { duration } from "./debug";
import { equal, flatten, NDArray, prod } from "./helpers";
import { View } from "./view";
import { Op, Add, Expand, Mul, Permute, Reshape, Sum, ReLU, Sign, Log, Neg, Reciprocal, Exp, Sub, Sin, Sqrt } from "./ops";
import { random } from "./random";

export class Tensor {
  readonly data: Float32Array;
  readonly view: View;
  grad?: Tensor;
  op?: Op;
  requiresGrad?: boolean;

  constructor(data: Float32Array, view: View, requiresGrad?: boolean, op?: Op, ) {
    this.data = data;
    this.view = view;
    this.op = op;
    this.requiresGrad = requiresGrad;
  }

  static from(ndArray: NDArray, requiresGrad?: boolean) {
    return new Tensor(new Float32Array(flatten(ndArray)), View.fromNDArray(ndArray), requiresGrad);
  }

  @duration()
  static rand(shape: number[], requiresGrad?: boolean) {
    const size = prod(shape);
    const data = new Float32Array(size);

    for (let i = 0; i < size; i++) data[i] = random.randomFloat();

    return new Tensor(data, new View(shape), requiresGrad);
  }

  @duration()
  static randn(shape: number[], requiresGrad?: boolean) {
    const u1 = Tensor.rand(shape);
    const u2 = Tensor.rand(shape);

    // From https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    const result = u1.log().mul(-2).sqrt().mul(u2.mul(Math.PI * 2).cos());

    result.requiresGrad = requiresGrad;

    return result;
  }

  @duration()
  static normal(shape: number[], mean = 0, std = 1, requiresGrad?: boolean) {
    const result = Tensor.randn(shape).mul(std).add(mean);

    result.requiresGrad = requiresGrad;

    return result;
  }
  
  @duration()
  static uniform(shape: number[], low = 0, high = 1, requiresGrad?: boolean) {
    const result = Tensor.rand(shape).mul(high - low);

    result.requiresGrad = requiresGrad;

    return result;
  }
  
  @duration()
  static kaimingUniform(shape: number[], a = 0.01, requiresGrad?: boolean) {
    const bound = Math.sqrt(3) * Math.sqrt(2 / (1 + a ** 2)) / Math.sqrt(prod(shape.slice(1)))
    
    return Tensor.uniform(shape, -bound, bound, requiresGrad);
  }

  @duration()
  static kaimingNormal(shape: number[], a = 0.01, requiresGrad?: boolean) {
    const std = Math.sqrt(2 / (1 + a ** 2)) / Math.sqrt(prod(shape.slice(1)))
    
    return Tensor.normal(shape, 0, std, requiresGrad);
  }

  @duration()
  static full(shape: number[], fillValue: number, requiresGrad?: boolean) {
    return new Tensor(new Float32Array(prod(shape)).fill(fillValue), new View(shape), requiresGrad);
  }

  static ones(shape: number[], requiresGrad?: boolean) {
    return Tensor.full(shape, 1, requiresGrad);
  }

  @duration()
  static zeros(shape: number[], requiresGrad?: boolean) {
    return new Tensor(new Float32Array(prod(shape)), new View(shape), requiresGrad);
  }

  onesLike(requiresGrad?: boolean) {
    return Tensor.ones(this.shape, requiresGrad);
  }

  zerosLike(requiresGrad?: boolean) {
    return Tensor.zeros(this.shape, requiresGrad);
  }

  get ndim() {
    return this.shape.length;
  }

  get shape() {
    return this.view.shape;
  }

  numel() {
    return prod(this.shape);
  }

  indices() {
    return this.view.indices();
  }

  get(indices: number[]) {
    if (indices.length !== this.ndim) throw new Error(`[Tensor.get] sub-tensor indexing is not supported for now`);

    const index = indices.reduce((acc, index, i) => acc + index * this.view.strides[i], 0);

    return this.data[index];
  }

  set(indices: number[], value: number) {
    if (indices.length !== this.ndim) throw new Error(`[Tensor.set] sub-tensor assignment is not supported for now`);

    const index = indices.reduce((acc, index, i) => acc + index * this.view.strides[i], 0);

    this.data[index] = value;
  }

  item() {
    if (this.numel() !== 1) throw new Error(`[Tensor.item] tensor has shape ${this.shape}`);

    return this.data[0];
  }

  permute(order: number[]) {
    const view = this.view.permute(order);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (this.requiresGrad) {
      result.op = new Permute(this, order);
      result.requiresGrad = true;
    }

    return result;
  }

  reshape(newShape: number[]) {
    const view = this.view.reshape(newShape);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (this.requiresGrad) {
      result.op = new Reshape(this);
      result.requiresGrad = true;
    }

    return result;
  }

  expand(newShape: number[]) {
    const view = this.view.expand(newShape);

    if (this.view === view) return this;

    const result = new Tensor(this.data, view);

    if (this.requiresGrad) {
      result.op = new Expand(this);
      result.requiresGrad = true;
    }

    return result;
  }

  transpose(dim0 = 1, dim1 = 0) {
    dim0 = dim0 < 0 ? this.ndim + dim0 : dim0;
    dim1 = dim1 < 0 ? this.ndim + dim1 : dim1;

    if (dim0 > this.ndim - 1) throw new Error(`[Tensor.transpose] dim0=${dim0} is out of bounds for tensor with ndim=${this.ndim}`);
    if (dim1 > this.ndim - 1) throw new Error(`[Tensor.transpose] dim1=${dim1} is out of bounds for tensor with ndim=${this.ndim}`);

    const order = this.shape.map((_, i) => i);
    order[dim0] = dim1;
    order[dim1] = dim0;

    return this.permute(order);
  }

  get T(): Tensor {
    return this.transpose();
  }

  private static broadcasted(a: Tensor, b: Tensor) {
    const maxDim = Math.max(a.shape.length, b.shape.length);
    const aShapeAligned = [...Array(maxDim - a.shape.length).fill(1), ...a.shape];
    const bShapeAligned = [...Array(maxDim - b.shape.length).fill(1), ...b.shape];

    const canBroadcast = aShapeAligned.every((ai, i) => ai === bShapeAligned[i] || ai === 1 || bShapeAligned[i] === 1);

    if (!canBroadcast) throw new Error(`[Tensor.broadcasted] shape values mismatch, can't broadcast ${a.shape} and ${b.shape}`);

    const shape = aShapeAligned.map((ai, i) => Math.max(ai, bShapeAligned[i]));

    return [a.reshape(aShapeAligned).expand(shape), b.reshape(bShapeAligned).expand(shape)];
  }

  @duration()
  dot(tensor: Tensor) {
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
  mul(value: Tensor | number) {
    const tensor = typeof value === "number" ? Tensor.from([value]) : value;

    const [a, b] = Tensor.broadcasted(this, tensor);

    const result = a.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, a.get(indices) * b.get(indices));
    }

    if (a.requiresGrad || b.requiresGrad) {
      result.op = new Mul(a, b);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  add(value: Tensor | number) {
    const tensor = typeof value === "number" ? Tensor.from([value]) : value;

    const [a, b] = Tensor.broadcasted(this, tensor);

    const result = a.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, a.get(indices) + b.get(indices));
    }

    if (a.requiresGrad || b.requiresGrad) {
      result.op = new Add(a, b);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  sub(value: Tensor | number) {
    const tensor = typeof value === "number" ? Tensor.from([value]) : value;

    const [a, b] = Tensor.broadcasted(this, tensor);

    const result = a.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, a.get(indices) - b.get(indices));
    }

    if (a.requiresGrad || b.requiresGrad) {
      result.op = new Sub(a, b);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  div(value: Tensor | number) {
    const tensor = typeof value === "number" ? Tensor.from([value]) : value;

    return this.mul(tensor.reciprocal());
  }

  @duration()
  sum(axis: number, keepDim = false) {
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

    if (this.requiresGrad) {
      result.op = new Sum(this, axis, keepDim);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  sign() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      const value = this.get(indices);

      result.set(indices, value > 0 ? 1 : value < 0 ? -1 : 0);
    }

    if (this.requiresGrad) {
      result.op = new Sign(this);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  relu() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      const value = this.get(indices);

      result.set(indices, value > 0 ? value : 0);
    }

    if (this.requiresGrad) {
      result.op = new ReLU(this);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  sqrt() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, Math.sqrt(this.get(indices)));
    }

    if (this.requiresGrad) {
      result.op = new Sqrt(this, result);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  sin() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, Math.sin(this.get(indices)));
    }

    if (this.requiresGrad) {
      result.op = new Sin(this);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  cos() {
    return Tensor.full(this.shape, Math.PI / 2).sub(this).sin();
  }

  @duration()
  log() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, Math.log(this.get(indices)));
    }

    if (this.requiresGrad) {
      result.op = new Log(this);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  exp() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, Math.exp(this.get(indices)));
    }

    if (this.requiresGrad) {
      result.op = new Exp(this, result);
      result.requiresGrad = true;
    }

    return result;
  }

  private softmaxParts() {
    const e = this.exp();
    const sum = e.sum(-1, true);

    return [e, sum];
  }

  @duration()
  softmax() {
    const [e, sum] = this.softmaxParts();

    return e.div(sum);
  }

  @duration()
  logSoftmax() {
    const [_, sum] = this.softmaxParts();

    return this.sub(sum.log());
  }

  @duration()
  sigmoid() {
    return this.neg().exp().add(this.onesLike()).reciprocal();
  }

  @duration()
  reciprocal() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, 1 / this.get(indices));
    }

    if (this.requiresGrad) {
      result.op = new Reciprocal(this, result);
      result.requiresGrad = true;
    }

    return result;
  }

  @duration()
  neg() {
    const result = this.zerosLike();

    for (let indices of result.indices()) {
      result.set(indices, -this.get(indices));
    }

    if (this.requiresGrad) {
      result.op = new Neg(this);
      result.requiresGrad = true;
    }

    return result;
  }

  private toposort() {
    const visit = (node: Tensor, visited: Set<Tensor>, sorted: Tensor[]) => {
      visited.add(node);

      if (node.op) {
        node.op.inputs.forEach((input) => {
          if (!visited.has(input)) {
            visit(input, visited, sorted);
          }
        });
      }

      sorted.push(node);

      return sorted;
    };

    return visit(this, new Set(), []);
  }

  backward(grad?: Tensor) {
    if (!this.requiresGrad)
      throw new Error(`[Tensor.backward] root Tensor has requiresGrad=${this.requiresGrad}`);

    grad = grad ?? this.onesLike();

    if (!equal(grad.shape, this.shape))
      throw new Error(`[Tensor.backward] backward grad has invalid shape: ${this.shape} !== ${grad.shape}`);

    this.grad = this.grad ? this.grad.add(grad) : grad;

    const sortedGraph = this.toposort();

    sortedGraph.reverse().forEach((node) => {
      if (!node.op) return;

      const inputGrads = node.op.backward(node.grad!);

      node.op.inputs.forEach((input, i) => {
        if (!input.requiresGrad) return;

        input.grad = input.grad ? input.grad.add(inputGrads[i]) : inputGrads[i];
      });
    });
  }

  detach() {
    return new Tensor(this.data, this.view, false);
  }

  render() {
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
