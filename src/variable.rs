use rand::prelude::*;

use crate::utils::max;

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    intrinsics::{cosf64, expf64, sinf64},
    ops::{Add, Deref, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Default, Clone)]
pub struct Variable(pub Rc<RefCell<VariableData>>);

#[derive(Debug)]
pub enum Operation {
    ADD,
    MUL,
    POW,
    Custom(String),
}

#[derive(Debug, Default)]
pub struct VariableData {
    pub data: f64,
    pub grad: f64,
    /// random id: u16. todo(...)
    pub id: u16,
    /// why not impl Fn(...) -> ... ?
    pub fun: Option<fn(&VariableData)>,
    pub op: Option<Operation>,
    pub children: Vec<Variable>,
}

impl Deref for Variable {
    type Target = Rc<RefCell<VariableData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().id == other.borrow().id
    }
}

impl Eq for Variable {}

impl Debug for Variable {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().id.hash(state)
    }
}

impl Add for &Variable {
    type Output = Variable;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Variable::from(self.borrow().data + rhs.borrow().data);
        out.borrow_mut().op = Some(Operation::ADD);
        out.borrow_mut().children = vec![Variable(Rc::clone(self)), Variable(Rc::clone(rhs))];

        out.borrow_mut().fun = Some(|x: &VariableData| {
            x.children[0].borrow_mut().grad += x.grad;
            x.children[1].borrow_mut().grad += x.grad;
        });

        out.borrow_mut().id = random();
        out
    }
}

impl Add<f64> for &Variable {
    type Output = Variable;

    fn add(self, rhs: f64) -> Self::Output {
        self + &Variable::from(rhs)
    }
}

impl Mul for &Variable {
    type Output = Variable;

    fn mul(self, rhs: &Variable) -> Self::Output {
        let out = Variable::from(self.borrow().data * rhs.borrow().data);
        out.borrow_mut().op = Some(Operation::MUL);
        out.borrow_mut().children = vec![Variable(Rc::clone(self)), Variable(Rc::clone(rhs))];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            let a = x.children[0].borrow().data;
            let b = x.children[1].borrow().data;
            x.children[0].borrow_mut().grad += b * x.grad;
            x.children[1].borrow_mut().grad += a * x.grad;
        });
        out
    }
}

impl Mul<f64> for &Variable {
    type Output = Variable;

    fn mul(self, rhs: f64) -> Self::Output {
        self * &Variable::from(rhs)
    }
}

impl Sub for &Variable {
    type Output = Variable;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Variable {
    pub fn new(tensor: VariableData) -> Variable {
        Variable(Rc::new(RefCell::new(tensor)))
    }

    pub fn from<T: Into<f64>>(f: T) -> Variable {
        let out = Variable::default();
        out.borrow_mut().data = f.into();
        out.borrow_mut().id = random();
        out
    }

    pub fn backward(&mut self) {
        self.borrow_mut().grad = 1.0;
        let (mut order, mut used) = (Vec::new(), HashSet::new());
        self.dfs(&mut order, &mut used);
        order.into_iter().rev().for_each(|it| {
            if let Some(fun) = it.borrow().fun {
                fun(&it.borrow());
            }
        });
    }

    pub fn dfs(&self, top_sort: &mut Vec<Variable>, used: &mut HashSet<Variable>) {
        if used.insert(self.clone()) {
            self.borrow().children.iter().for_each(|child| {
                child.dfs(top_sort, used);
            });
            top_sort.push(self.clone());
        }
    }

    pub fn pow(&self, _power: f64) -> Variable {
        let out = Variable::from(self.borrow().data.powf(_power));
        out.borrow_mut().op = Some(Operation::POW);
        out.borrow_mut().children = vec![self.clone(), Variable::from(_power)];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            // w -> ... -> x -> y (y = x^p) -> ... -> L
            // dL/dx = dL/dy * dy/dx = dL/dy * p * x^(p-1)
            let pow = x.children[1].borrow().data;
            let a = x.children[0].borrow().data.powf(pow - 1.0) * pow;
            x.children[0].borrow_mut().grad += x.grad * a;
        });
        out.borrow_mut().id = random();
        out
    }

    pub fn sin(&self) -> Variable {
        let out = Variable::from(unsafe { sinf64(self.borrow().data) });
        out.borrow_mut().op = Some(Operation::Custom(String::from("sin")));
        out.borrow_mut().children = vec![self.clone()];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            let val = x.children[0].borrow().data;
            x.children[0].borrow_mut().grad += x.grad * unsafe { cosf64(val) };
        });
        out
    }

    pub fn cos(&self) -> Variable {
        let out = Variable::from(unsafe { cosf64(self.borrow().data) });
        out.borrow_mut().op = Some(Operation::Custom(String::from("cos")));
        out.borrow_mut().children = vec![self.clone()];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            let val = x.children[0].borrow().data;
            x.children[0].borrow_mut().grad += -x.grad * unsafe { sinf64(val) };
        });
        out
    }

    pub fn relu(&self) -> Variable {
        // x -> y (=relu x) -> L
        // dL/dx = dL/dy * dy/dx = dL/dy * I[x>0]
        let out = Variable::from(max(self.borrow().data, 0.0));
        out.borrow_mut().op = Some(Operation::Custom(String::from("relu")));
        out.borrow_mut().children = vec![self.clone()];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            if x.children[0].borrow().data > 0.0 {
                // x.grad * I[x > 0]
                x.children[0].borrow_mut().grad += x.grad;
            }
        });
        out
    }

    pub fn exp(&self) -> Variable {
        let exp = unsafe { expf64(self.borrow().data) };
        let out = Variable::from(exp);
        out.borrow_mut().op = Some(Operation::Custom(String::from("exp")));
        out.borrow_mut().children = vec![self.clone()];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            // x -> y (=exp(x)) -> L
            // dL/dx=dL/dy * dy/dx
            let val = unsafe { expf64(x.children[0].borrow_mut().data) };
            x.children[0].borrow_mut().grad += x.grad * val;
        });
        out
    }

    pub fn silu(&self) -> Variable {
        let out = self * &self.sigmoid();
        out.borrow_mut().op = Some(Operation::Custom(String::from("silu"))); // todo
        out
    }

    pub fn sigmoid(&self) -> Variable {
        // x -> 1 / (1 + exp(-x))
        // maybe faster?
        let out = (&Variable::from(1.0) + &(-self).exp()).pow(-1.0);
        out.borrow_mut().op = Some(Operation::Custom(String::from("sigmoid"))); // is it useful?
        out
    }
}
