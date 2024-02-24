use std::{
    cell::RefCell,
    ops::{Deref, Mul},
    rc::Rc,
};

use crate::Variable;

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
    fn new(r: usize, c: usize) -> Tensor2D {
        let v = vec![vec![Variable::default(); c]; r];
        Tensor2D(Rc::new(RefCell::new(v)), (r, c))
    }
}

impl Mul<&Tensor2D> for &Tensor1D {
    type Output = Tensor1D;

    fn mul(self, rhs: &Tensor2D) -> Self::Output {
        let out = Tensor1D::new(rhs.1 .1);
        for i in 0..rhs.1 .1 {
            let mut c = Variable::from(0.0);
            for k in 0..rhs.1 .0 {
                c = &c + &(&self.0.borrow()[k] * &rhs.0.borrow()[k][i]);
            }
            out.0.borrow_mut()[i] = c;
        }
        out
    }
}

impl Tensor1D {
    pub fn t(&self) -> Tensor2D {
        let out = Tensor2D::new(self.1, 1);
        for i in 0..self.1 {
            out.0.borrow_mut()[i][0] = self.borrow()[i].clone();
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
}
