use crate::{Operation, Variable, VariableData};
use auto_ops::*;
use std::{cell::RefCell, ops::Deref, rc::Rc};

/// Tensor1D is R^n. 1 column!
#[derive(Default)]
pub struct Tensor1D(pub Rc<RefCell<Vec<Variable>>>, pub usize);

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

impl Tensor2D {
    pub fn new(r: usize, c: usize) -> Tensor2D {
        // we can't use vec![T::default(), ...], because of cloning Rc...
        // todo: something smarter?
        let v = (0..r)
            .map(|_| (0..c).map(|_| Variable::default()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Tensor2D(Rc::new(RefCell::new(v)), (r, c))
    }

    pub fn from(v: &Vec<Vec<f64>>) -> Tensor2D {
        // todo ...
        let out = Self::new(v.len(), v[0].len());
        for i in 0..(v.len()) {
            for j in 0..(v[0].len()) {
                out.borrow_mut()[i][j].0.borrow_mut().data = v[i][j];
            }
        }
        out
    }
}

// Tensor operations
impl_op_ex!(*|a: &Tensor2D, b: &Tensor1D| -> Tensor1D {
    // todo: refactor
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

impl Tensor1D {
    pub fn new(n: usize) -> Tensor1D {
        Tensor1D(
            Rc::new(RefCell::new(
                (0..n).map(|_| Variable::default()).collect::<Vec<_>>(),
            )),
            n,
        )
    }

    pub fn from(v: &Vec<f64>) -> Tensor1D {
        // todo ...
        let out = Self::new(v.len());
        for i in 0..(v.len()) {
            out.borrow_mut()[i].0.borrow_mut().data = v[i];
        }
        out
    }

    pub fn t(&self) -> Tensor2D {
        // todo: refactor, how to avoid using .borrow_mut() each time?
        let out = Tensor2D::new(1, self.1);
        for i in 0..self.1 {
            out.0.borrow_mut()[0][i] = self.borrow()[i].clone();
        }
        out
    }

    pub fn cast(&self) -> Variable {
        assert!(self.1 == 1);
        self.0.borrow()[0].clone()
    }

    pub fn backward(&self) {
        self.cast().backward();
    }

    pub fn apply_fn(&self, fun: fn(&Variable) -> Variable) -> Tensor1D {
        // todo: refactor. how to avoid using phantom_data ?
        let y = Tensor1D::new(self.1);
        let mut interior_new = y.borrow_mut();
        let interior_prev = self.0.borrow_mut();
        for i in 0..(self.1) {
            interior_new[i] = fun(&interior_prev[i]);
        }
        drop(interior_new);
        y
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

    pub fn sum(&self) -> Variable {
        // we don't want this behaviour:
        // a + b + c + d + e <-> x = a + b, y = x + c, z = y + d, t = z + e
        // grad graph will be very long

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
}
