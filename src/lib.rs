#![allow(dead_code)]
#[warn(unused_variables)]
use rand::prelude::*;
use std::collections::HashSet;

use std::hash::Hash;
use std::ops::{Add, Deref};
use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Default, Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

#[derive(Debug)]
pub enum Operation {
    ADD,
    MUL,
    POW,
}

#[derive(Debug, Default)]
pub struct TensorData {
    pub data: f64,
    pub grad: f64,
    /// random id: u16. todo(...)
    pub id: u16,
    pub fun: Option<fn(&TensorData)>, // why not impl Fn(...) -> ... ?
    pub op: Option<Operation>,
    pub children: Vec<Tensor>,
}

impl Tensor {
    fn from<T: Into<f64>>(f: T) -> Tensor {
        let out = Tensor::default();
        out.borrow_mut().data = f.into();
        out.borrow_mut().id = random();
        out
    }
}

impl Deref for Tensor {
    type Target = Rc<RefCell<TensorData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().id == other.borrow().id
    }
}

impl Eq for Tensor {}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().id.hash(state)
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Tensor::from(self.borrow().data + rhs.borrow().data);
        out.borrow_mut().op = Some(Operation::ADD);
        out.borrow_mut().children = vec![Tensor(Rc::clone(self)), Tensor(Rc::clone(rhs))];

        out.borrow_mut().fun = Some(|x: &TensorData| {
            x.children[0].borrow_mut().grad += x.grad;
            x.children[1].borrow_mut().grad += x.grad;
        });

        out.borrow_mut().id = random();
        out
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        self + &Tensor::from(rhs)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let out = Tensor::from(self.borrow().data * rhs.borrow().data);
        out.borrow_mut().op = Some(Operation::MUL);
        out.borrow_mut().children = vec![Tensor(Rc::clone(self)), Tensor(Rc::clone(rhs))];
        out.borrow_mut().fun = Some(|x: &TensorData| {
            let a = x.children[0].borrow().data;
            let b = x.children[1].borrow().data;
            x.children[0].borrow_mut().grad += b * x.grad;
            x.children[1].borrow_mut().grad += a * x.grad;
        });
        out
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        self * &Tensor::from(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Tensor {
    fn new(tensor: TensorData) -> Tensor {
        Tensor(Rc::new(RefCell::new(tensor)))
    }

    fn backward(&mut self) {
        self.borrow_mut().grad = 1.0;
        let (mut order, mut used) = (Vec::new(), HashSet::new());
        self.dfs(&mut order, &mut used);
        order.into_iter().rev().for_each(|it| {
            if let Some(fun) = it.borrow().fun {
                fun(&it.borrow());
            }
        });
    }

    pub fn pow(&self, _power: f64) -> Tensor {
        let out = Tensor::from(self.borrow().data.powf(_power));
        out.borrow_mut().op = Some(Operation::POW);
        out.borrow_mut().children = vec![self.clone()];
        // w -> ... -> x -> y (y = x^p) -> ... -> L
        // dL/dx = dL/dy * dy/dx = dL/dy * p * x^(p-1)
        out.borrow_mut().fun = Some(|x: &TensorData| {
            let pow = x.children[1].borrow().data;
            x.children[0].borrow_mut().grad +=
                x.grad * x.children[0].borrow().data.powf(pow - 1.0) * pow;
        });
        out.borrow_mut().id = random();
        out
    }

    pub fn dfs(&self, top_sort: &mut Vec<Tensor>, used: &mut HashSet<Tensor>) {
        if used.insert(self.clone()) {
            self.borrow().children.iter().for_each(|child| {
                child.dfs(top_sort, used);
            });
            top_sort.push(self.clone());
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn construct() {
        let id_ = random();
        let data = TensorData {
            data: 2.0,
            grad: 0.0,
            id: id_,
            fun: None,
            op: Some(Operation::ADD),
            children: Vec::new(),
        };

        let tensor = Tensor::new(data);
        assert_eq!(tensor.0.borrow().id, id_);
        let s = &tensor * 3.;
        let m = &tensor + 6.;
        assert_eq!(s.0.borrow().data, 6.);
        assert_eq!(m.0.borrow().data, 8.);
    }

    #[test]
    fn simple_grad_add() {
        let a = Tensor::from(3.);
        let b = Tensor::from(4.);
        let mut c = &a + &b; // why &mut ?
        c.backward();
        assert_eq!(a.borrow().grad, 1.0);
    }

    #[test]
    fn simple_grad_add_self() {
        let a = Tensor::from(3.);
        let mut c = &a + &a;
        c.backward();
        assert_eq!(a.borrow().grad, 2.0);
    }

    #[test]
    fn simple_grad_mul() {
        let a = Tensor::from(3.);
        let b = Tensor::from(4.);
        let mut c = &a * &b;
        c.backward();
        assert_eq!(a.borrow().grad, 4.0);
        assert_eq!(b.borrow().grad, 3.0);
    }

    #[test]
    fn simple_grad_self_mul() {
        let a = Tensor::from(3.);
        let mut c = &a * &a;
        c.backward();
        assert_eq!(a.borrow().grad, 6.0);
    }
}
