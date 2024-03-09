# rust-minigrad

Rust implementation of a miniature automatic differentiation library

`example:`
```rs
let x = Variable::from(2.0);
let mut y = (&x * &x).tan();
y.backward();
assert_close!(x.grad(), 9.3622, 0.001);
```
