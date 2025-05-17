// LCG-based pseudorandom number generator
// https://stackoverflow.com/questions/424292/seedable-javascript-random-number-generator

const m = 0x80000000; // 2**31;
const a = 1103515245;
const c = 12345;

let state = Math.floor(Math.random() * (m - 1));

/** Returns pseudorandom int */
const randomInt = () => {
  state = (a * state + c) % m;

  return state;
};

/** Returns pseudorandom float in range [0, 1) */
const randomFloat = () => randomInt() / (m - 1);

const seed = (value: number) => {
  state = value;
};

export const random = {
  seed,
  randomInt,
  randomFloat,
};
