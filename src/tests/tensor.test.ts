import { Tensor } from "../tensor";

describe("add", () => {
  test("forward and backward", () => {
    const a = Tensor.from([
      [1, 2, 3],
      [5, 7, 11],
    ]);

    const b = Tensor.from([
      [13, 17, 19],
      [23, 29, 31],
    ]);

    const expected = Tensor.from([
      [14, 19, 22],
      [28, 36, 42],
    ]);

    const result = a.add(b);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(a.onesLike().render());
    expect(b.grad?.render()).toEqual(b.onesLike().render());
  });
});

describe("mul", () => {
  test("forward and backward", () => {
    const a = Tensor.from([
      [1, 2, 3],
      [5, 7, 11],
    ]);

    const b = Tensor.from([
      [13, 17, 19],
      [23, 29, 31],
    ]);

    const expected = Tensor.from([
      [13, 34, 57],
      [115, 203, 341],
    ]);

    const result = a.mul(b);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(b.render());
    expect(b.grad?.render()).toEqual(a.render());
  });
});

describe("reshape", () => {
  test("forward and backward", () => {
    const a = Tensor.from([
      [1, 2, 3, 5],
      [7, 11, 13, 17],
    ]);

    const expected = Tensor.from([
      [
        [1, 2],
        [3, 5],
      ],
      [
        [7, 11],
        [13, 17],
      ],
    ]);

    const result = a.reshape([2, 2, 2]);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(a.onesLike().render());
  });
});

describe("permute", () => {
  test("forward and backward", () => {
    const a = Tensor.from([
      [1, 2, 3, 5],
      [7, 11, 13, 17],
    ]);

    const expected = Tensor.from([
      [1, 7],
      [2, 11],
      [3, 13],
      [5, 17],
    ]);

    const result = a.permute([1, 0]);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(a.onesLike().render());
  });
});

describe("expand", () => {
  test("forward and backward", () => {
    const a = Tensor.from([[[1, 2, 3, 5]], [[7, 11, 13, 17]]]);

    const expected = Tensor.from([
      [
        [1, 2, 3, 5],
        [1, 2, 3, 5],
        [1, 2, 3, 5],
      ],
      [
        [7, 11, 13, 17],
        [7, 11, 13, 17],
        [7, 11, 13, 17],
      ],
    ]);

    const result = a.expand([2, 3, 4]);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(Tensor.full([2, 1, 4], 3).render());
  });
});

describe("sum", () => {
  const getInput = () => Tensor.from([
    [
      [13, 58, 129],
      [17, 62, 141],
      [19, 74, 159],
      [23, 82, 177],
    ],
    [
      [65, 203, 473],
      [85, 217, 517],
      [95, 259, 583],
      [115, 287, 649],
    ],
  ]);

  test("forward and backward, keepDim=false", () => {
    const a = getInput();

    const expected = Tensor.from([
      [200, 220, 252, 282],
      [741, 819, 937, 1051],
    ]);

    const result = a.sum(-1, undefined, false);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(a.onesLike().render());
  });

  test("forward and backward, keepDim=true", () => {
    const a = getInput();

    const expected = Tensor.from([
      [[200], [220], [252], [282]],
      [[741], [819], [937], [1051]],
    ]);

    const result = a.sum(-1, undefined, true);
    expect(result.render()).toEqual(expected.render());

    result.backward();
    expect(a.grad?.render()).toEqual(a.onesLike().render());
  });
});

describe("dot", () => {
  test("forward", () => {
    const a = Tensor.from([
      [1, 2, 3],
      [5, 7, 11],
    ]);

    const b = Tensor.from([
      [13, 17, 19, 23],
      [29, 31, 37, 41],
      [43, 47, 53, 59],
    ]);

    const expected = Tensor.from([
      [200, 220, 252, 282],
      [741, 819, 937, 1051],
    ]);

    const result = a.dot(b);

    expect(result.render()).toEqual(expected.render());
  });
});
