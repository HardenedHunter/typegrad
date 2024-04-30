# autograd-ts

## What is this?

A simple deep learning library with React-powered UI for debugging.

## What is the goal of this project?

- better understand how autograd engines work under the hood
- better understand [tinygrad](https://github.com/tinygrad/tinygrad)'s ShapeTracker, laziness and movement ops
- build something that works and can be used for training and inference

## How do we get there? Definition-of-Done for MVP:

- create an autograd engine (based on Karpathy's [micrograd](https://github.com/karpathy/micrograd))
- create a simple Tensor library with PyTorch/tinygrad/..-like frontend
- implement basic operations, like `.dot(...)` and others
- get reasonable results for MNIST with both training and inference (e. g. the model actually converges to sub-90% on test set)
- visualize everything when core primitives are ready
- ..and do all of the above in TypeScript because we like to suffer

## If we succeed, try to do the following

- implement all movement ops (reshape, expand, shrink, permute, flip)
- support for Conv2D (should be easy when movement ops are ready)
- tune for performance
- maybe add support for fast backends (Metal API?)
- loader for ONNX models

## Why TypeScript?

It would be pretty straightforward to do this with Python or C++. Hovewer, there are countless projects built with those languages to express the same idea/solve the same set of problems. To put it simply, we will try to push TypeScript to it's limit and see what we can do.

Pretty much the only major downside of TS for this particular project is lack of first-class support for operator overloading:

```ts
const a = new Tensor([1, 2, 3]);
const b = new Tensor([4, 5, 6]);
const c = a + b; // Sadly, we can't do that
const c1 = a.add(b); // ..but we can do this
const c2 = add(a, b); // ..or this
```

This doesn't look that bad! But here's another one:

```ts
const c = a * 2 + (1 - b); // Looks good
const c = add(mul(a, 2), sub(1, b)); // Looks terrible!
```

This ultimately leads to bloated and less readable code, but since we are here to do Deep Learning, most operations will be hidden inside library code anyway, and high-level code that does training or inference will rarely use so many operations inline.

TypeScript also happens to be the language I have the most experience with. So as I stated previously, I will try to do my best and push the language to it's limit.
