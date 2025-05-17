import { equal, getShapeOfNDArray, NDArray, prod } from "./helpers";

const canonicalizeStrides = (shape: number[], strides: number[]) => strides.map((stride, i) => (shape[i] === 1 ? 0 : stride));

const stridesForShape = (shape: number[]): number[] => {
  const size = shape.length;
  const strides = Array<number>(size).fill(0);

  strides[size - 1] = 1;

  for (let i = size - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  return canonicalizeStrides(shape, strides);
};

export class View {
  public shape: number[];
  public strides: number[];
  public contiguous: boolean;

  public constructor(shape: number[], strides?: number[]) {
    const defaultStrides = stridesForShape(shape);

    this.shape = shape;
    this.strides = strides ? canonicalizeStrides(shape, strides) : defaultStrides;
    this.contiguous = equal(this.strides, defaultStrides);
  }

  public static fromNDArray(ndArray: NDArray) {
    return new View(getShapeOfNDArray(ndArray));
  }

  public get ndim() {
    return this.shape.length;
  }

  public permute(order: number[]) {
    if (this.ndim !== order.length) throw new Error(`[View.permute] order length is invalid: ${order.length} !== ${this.ndim}`);

    const shape = this.shape.map((_, i) => this.shape[order[i]]);

    if (equal(shape, this.shape)) return this;

    const strides = this.strides.map((_, i) => this.strides[order[i]]);

    return new View(shape, strides);
  }

  public reshape(newShape: number[]) {
    if (equal(newShape, this.shape)) return this;

    const currentSize = prod(this.shape);
    const newSize = prod(newShape);

    if (currentSize !== newSize) throw new Error(`[View.reshape] size mismatched, can't reshape ${this.shape} -> ${newShape}`);

    if (this.contiguous) return new View(newShape);

    throw new Error("[View.reshape] reshape of non-contiguous view is currently not supported");
  }

  public expand(newShape: number[]) {
    if (equal(newShape, this.shape)) return this;

    if (this.ndim !== newShape.length) throw new Error(`[View.expand] shape length mismatch, can't expand ${this.shape} -> ${newShape}`);

    if (!this.shape.every((s, i) => s === newShape[i] || s === 1))
      throw new Error(`[View.expand] shape values mismatch, can't expand ${this.shape} -> ${newShape}`);

    return new View(newShape, this.strides);
  }

  public *indices() {
    const size = prod(this.shape);
    const indices = new Array(this.ndim).fill(0);

    for (let i = 0; i < size; i++) {
      yield [...indices];

      let axis = this.ndim - 1;

      while (axis >= 0 && indices[axis] + 1 >= this.shape[axis]) {
        indices[axis] = 0;
        axis--;
      }

      if (axis >= 0) indices[axis]++;
    }
  }
}
