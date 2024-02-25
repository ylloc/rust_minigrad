use crate::Variable;
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

impl Tensor2D {
    pub fn new(r: usize, c: usize) -> Tensor2D {
        let v = vec![vec![Variable::default(); c]; r];
        Tensor2D(Rc::new(RefCell::new(v)), (r, c))
    }
}

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
    pub fn t(&self) -> Tensor2D {
        // todo: refactor
        let out = Tensor2D::new(1, self.1);
        for i in 0..self.1 {
            out.0.borrow_mut()[0][i] = self.borrow()[i].clone();
        }
        out
    }

    pub fn new(n: usize) -> Tensor1D {
        Tensor1D(Rc::new(RefCell::new(vec![Variable::default(); n])), n)
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
}
