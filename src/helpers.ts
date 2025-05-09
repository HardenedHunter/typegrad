export type NDArray = Array<number> | Array<NDArray>;

export const flatten = (data: NDArray): number[] =>
  data.reduce<number[]>((acc, item) => (typeof item === "number" ? [...acc, item] : [...acc, ...flatten(item)]), []);

export const getShapeOfNDArray = (data: NDArray, strict?: boolean): number[] => {
  if (data.length === 0) throw new Error("[getShapeOfRawData] data.length === 0");

  if (strict) {
    const leaf = data.every((item) => typeof item === "number");
    const nested = data.every((item) => item instanceof Array);

    if (!leaf && !nested) throw new Error(`[getShapeOfRawData] uneven array: ${data}`);
  }

  if (typeof data[0] === "number") {
    return [data.length];
  }

  return [data.length, ...getShapeOfNDArray(data[0], strict)];
};

export const equal = (a: number[], b: number[]) => a.every((ai, i) => ai === b[i]);

export const prod = (data: number[]) => data.reduce((acc, item) => acc * item, 1);

/**
 * Same as `numpy.argsort`. Returns the indices that would sort the array.
 * https://numpy.org/doc/2.2/reference/generated/numpy.argsort.html
 */
export const argsort = (data: number[]) => {
  return data.map((item, i) => [item, i]).sort((a, b) => a[0] - b[0]).map((a) => a[1]);
};
