use crate::{Operation, Variable, VariableData};
use auto_ops::*;
use std::{cell::RefCell, ops::Deref, rc::Rc};

/// Equvalent of R^d. One column
#[derive(Default)]
pub struct Tensor1D(pub Rc<RefCell<Vec<Variable>>>, pub usize);

/// Equvalent to matrix: R^(a * b)
#[derive(Default)]
pub struct Tensor2D(pub Rc<RefCell<Vec<Vec<Variable>>>>, pub (usize, usize));

impl Deref for Tensor1D {
    type Target = Rc<RefCell<Vec<Variable>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Tensor2D {
    type Target = Rc<RefCell<Vec<Vec<Variable>>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl_op_ex!(*|a: &Tensor2D, b: &Tensor1D| -> Tensor1D {
    assert_eq!(a.1 .1, b.1);
    let out = Tensor1D::new(a.1 .0);
    for i in 0..(a.1 .0) {
        let mut c = Variable::from(0.0);
        for k in 0..(a.1 .1) {
            c = c + &a.0.borrow()[i][k] * &b.0.borrow()[k];
        }
        out.0.borrow_mut()[i] = c;
    }
    out
});

impl_op_ex!(*|a: &Tensor2D, b: &Tensor2D| -> Tensor2D {
    assert_eq!(a.1 .1, b.1 .0);
    let out = Tensor2D::new(a.1 .0, b.1 .1);
    for i in 0..(a.1 .0) {
        for j in 0..(b.1 .1) {
            let mut c = Variable::from(0.0);
            for k in 0..(a.1 .1) {
                c = c + &a.0.borrow()[i][k] * &b.0.borrow()[k][j];
            }
            out.0.borrow_mut()[i][j] = c;
        }
    }
    out
});

impl_op_ex!(*|a: &Tensor2D, b: &Variable| -> Tensor2D {
    let out = Tensor2D::new(a.1 .0, a.1 .1);
    for i in 0..(a.1 .0) {
        for j in 0..(a.1 .1) {
            out.borrow_mut()[i][j] = &a.0.borrow()[i][j] * b;
        }
    }
    out
});

// +
//
impl_op_ex!(*|a: &Variable, b: &Tensor2D| -> Tensor2D { b * a });

impl_op_ex!(+|a: &Tensor2D, b: &Tensor2D| -> Tensor2D {
    assert_eq!(a.1, b.1);
    let out = Tensor2D::new(a.1 .0, b.1 .1);
    for i in 0..(a.1 .0) {
        for j in 0..(a.1 .1) {
            out.0.borrow_mut()[i][j] = &a.0.borrow()[i][j] * &b.0.borrow()[j][j];
        }
    }
    out
});

impl_op_ex!(+|a: &Tensor2D, b: &Variable| -> Tensor2D {
    let out = Tensor2D::new(a.1 .0, a.1 .1);
    for i in 0..(a.1 .0) {
        for j in 0..(a.1 .1) {
            let x = &a.borrow()[i][j];
            out.borrow_mut()[i][j] = x + b;
        }
    }
    out
});

impl_op_ex!(+|a: &Variable, b: &Tensor2D| -> Tensor2D { b + a });

impl_op_ex!(-|a: &Tensor2D, b: &Variable| -> Tensor2D { a + -b });
impl_op_ex!(-|a: &Tensor2D| -> Tensor2D {
    let out = Tensor2D::new(a.1 .0, a.1 .1);
    for i in 0..(a.1 .0) {
        for j in 0..(a.1 .1) {
            let x = &a.borrow()[i][j];
            out.borrow_mut()[i][j] = x * -1.;
        }
    }
    out
});

impl_op_ex!(-|a: &Variable, b: &Tensor2D| -> Tensor2D { a + -b });

impl_op_ex!(/|a: &Tensor2D, b: &Variable| -> Tensor2D { a * (1.0 / b) });

impl_op_ex!(*|a: &Tensor1D, b: &Variable| -> Tensor1D {
    let out = Tensor1D::new(a.1);
    for i in 0..(a.1) {
        let x = &a.borrow()[i];
        out.borrow_mut()[i] = x * b;
    }
    out
});

impl_op_ex!(*|a: &Variable, b: &Tensor1D| -> Tensor1D { b * a });
impl_op_ex!(/|a: &Tensor1D, b: &Variable| -> Tensor1D { a * (1.0 / b) });

impl_op_ex!(+|a: &Tensor1D, b: &Variable| -> Tensor1D {
    let out = Tensor1D::new(a.1);
    for i in 0..(a.1) {
        let x = &a.borrow()[i];
        out.borrow_mut()[i] = x + b;
    }
    out
});

impl_op_ex!(+|a: &Variable, b: &Tensor1D| -> Tensor1D { b + a });

// TODO: use impl_op_commutative!()

impl Tensor1D {
    pub fn new(n: usize) -> Tensor1D {
        Tensor1D(
            Rc::new(RefCell::new(
                (0..n).map(|_| Variable::default()).collect::<Vec<_>>(),
            )),
            n,
        )
    }

    pub fn shape(&self) -> (usize,) {
        (self.1,)
    }

    pub fn from(v: &Vec<f64>) -> Tensor1D {
        assert!(!v.is_empty(), "can't create empty tensor2D");

        let out = Self::new(v.len());
        out.borrow_mut()
            .iter_mut()
            .zip(v.iter())
            .for_each(|(o, &v)| {
                o.borrow_mut().data = v;
            });

        out
    }

    /// transforms Tensor1D(_, d) to Tensor2D(_, (1, d))
    pub fn t(&self) -> Tensor2D {
        let out = Tensor2D::new(1, self.1);
        out.borrow_mut()[0]
            .iter_mut()
            .zip(self.borrow().iter())
            .for_each(|(o, v)| {
                *o = v.clone();
            });
        out
    }

    /// If Tensor1D represents a single Variable, it can be casted
    pub fn cast(&self) -> Variable {
        assert_eq!(
            self.1, 1,
            "Tensor1D must have exactly one element to be casted to Variable."
        );
        self.0.borrow()[0].clone()
    }

    /// performs backward pass if the tensor is equavalent to a Variable
    pub fn backward(&self) {
        self.cast().backward();
    }

    pub fn apply_fn(&self, fun: fn(&Variable) -> Variable) -> Tensor1D {
        let out = Tensor1D::new(self.1);
        let current = self.0.borrow();

        // Apply the function to each element and fill the output tensor
        out.borrow_mut()
            .iter_mut()
            .zip(current.iter())
            .for_each(|(o, v)| {
                *o = fun(v);
            });

        out
    }

    pub fn sin(&self) -> Tensor1D {
        self.apply_fn(|x| x.sin())
    }

    pub fn cos(&self) -> Tensor1D {
        self.apply_fn(|x| x.cos())
    }

    pub fn silu(&self) -> Tensor1D {
        self.apply_fn(|x| x.silu())
    }

    pub fn relu(&self) -> Tensor1D {
        self.apply_fn(|x| x.relu())
    }

    pub fn exp(&self) -> Tensor1D {
        self.apply_fn(|x| x.exp())
    }

    /// returns (Variable) - the sum of all interior elements
    pub fn sum(&self) -> Variable {
        // We want to avoid creating a long graph.
        // So tensor elements will be the childen of the resulting node.
        let out = Variable::from(self.borrow().iter().map(|x| x.data()).sum::<f64>());
        out.borrow_mut().op = Some(Operation::Custom(String::from("sum1D")));
        out.borrow_mut().children = self.borrow().clone();
        out.borrow_mut().fun = Some(|x: &VariableData| {
            let grad = x.grad;
            x.children.iter().for_each(|child| {
                child.borrow_mut().grad += grad;
            })
        });

        out
    }

    pub fn mean(&self) -> Variable {
        self.sum() / (self.1 as f64)
    }

    pub fn softmax(&self) -> Tensor1D {
        let x = self.clone().exp();
        let y = x.sum();
        x / y
    }

    /// -p * ln p
    pub fn cross_entropy_loss(&self, other: &Tensor1D) -> Variable {
        assert!(
            self.shape() == other.shape(),
            "1D shapes must be equal to use it"
        );
        unimplemented!()
    }
}

impl Tensor2D {
    /// Resurns default matrix, filled with zero.
    ///
    /// We can't actually use vec![T::default(); _] here, because of cloning rc.
    ///
    /// todo: something smarter?
    pub fn new(r: usize, c: usize) -> Tensor2D {
        let v = (0..r)
            .map(|_| (0..c).map(|_| Variable::default()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Tensor2D(Rc::new(RefCell::new(v)), (r, c))
    }

    pub fn shape(&self) -> (usize, usize) {
        self.1
    }

    pub fn from(v: &Vec<Vec<f64>>) -> Tensor2D {
        assert!(!v.is_empty(), "can't create empty tensor2D");
        let (r, c) = (v.len(), v[0].len());
        let out = Self::new(r, c);
        // trying to avoid double `.borrow_mut()` every step.
        // {  ...  } - to make compiler happy :)
        {
            let inner = out.borrow_mut();
            for i in 0..r {
                for j in 0..c {
                    inner[i][j].borrow_mut().data = v[i][j];
                }
            }
        }

        out
    }

    pub fn sum(&self) -> Variable {
        // Same strategy as in Tensor1D
        let out = Variable::from(
            self.borrow()
                .iter()
                .flatten()
                .map(|x| x.data())
                .sum::<f64>(),
        );
        out.borrow_mut().op = Some(Operation::Custom(String::from("sum2D")));
        out.borrow_mut().children = self.borrow().iter().flatten().cloned().collect::<Vec<_>>();
        out.borrow_mut().fun = Some(|x: &VariableData| {
            let grad = x.grad;
            x.children.iter().for_each(|child| {
                child.borrow_mut().grad += grad;
            })
        });

        out
    }

    pub fn mean(&self) -> Variable {
        self.sum() / ((self.1 .0) * (self.1 .1)) as f64
    }
}
