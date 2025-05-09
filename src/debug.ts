type Color = "green" | "magenta";

const colorCodes: { [key in Color]: number } = {
  green: 32,
  magenta: 35,
};

const addColor = (color: Color, s: string) => `\x1b[${colorCodes[color]}m${s}\x1b[0m`;

const MEASURE_ALL = false;

export const duration =
  (only?: boolean) =>
  <T extends object | Function>(target: T, propertyKey: string, descriptor: PropertyDescriptor) => {
    if (!MEASURE_ALL && !only) return descriptor;

    const originalMethod = descriptor.value;
    const className = target instanceof Function ? target.name : target.constructor.name;

    const methodName = addColor("magenta", `[${className}.${propertyKey}]`);

    descriptor.value = function (...args: unknown[]) {
      const startTime = performance.now();
      const result = originalMethod.apply(this, args);
      const endTime = performance.now();
      const timespan = endTime - startTime;

      const time = addColor("green", `${timespan.toFixed(3)} ms`);

      console.log(`${methodName} ${time}`);

      return result;
    };

    return descriptor;
  };
