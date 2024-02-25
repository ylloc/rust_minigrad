use crate::utils::max;
use auto_ops::*;
use rand::prelude::*;

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    intrinsics::{cosf64, expf64, sinf64},
    ops::Deref,
    rc::Rc,
};

#[derive(Default, Clone)]
pub struct Variable(pub Rc<RefCell<VariableData>>);

#[derive(Debug, Clone)]
pub enum Operation {
    ADD,
    MUL,
    POW,
    DIV,
    Custom(String),
    None,
}

#[derive(Default)]
pub struct VariableData {
    pub data: f64,
    pub grad: f64,
    pub id: u16,
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

impl Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().id.hash(state)
    }
}

// Operations for variables. auto_ops is amazing!
impl_op_ex!(+ |a: &Variable, b: &Variable| -> Variable {
    let out = Variable::from(a.borrow().data + b.borrow().data);
    out.borrow_mut().op = Some(Operation::ADD);
    out.borrow_mut().children = vec![Variable(Rc::clone(a)), Variable(Rc::clone(b))];
    out.borrow_mut().fun = Some(|x: &VariableData| {
        x.children[0].borrow_mut().grad += x.grad;
        x.children[1].borrow_mut().grad += x.grad;
    });
    out.borrow_mut().id = random();
    out
});

impl_op_ex!(+|a: &Variable, b: f64| -> Variable { a + Variable::from(b) });
impl_op_ex!(+|a: f64, b: &Variable| -> Variable { b + Variable::from(a) });

impl_op_ex!(*|a: &Variable, b: &Variable| -> Variable {
    let out = Variable::from(a.borrow().data * b.borrow().data);
    out.borrow_mut().op = Some(Operation::MUL);
    out.borrow_mut().children = vec![Variable(Rc::clone(a)), Variable(Rc::clone(b))];
    out.borrow_mut().fun = Some(|x: &VariableData| {
        let a = x.children[0].borrow().data;
        let b = x.children[1].borrow().data;
        x.children[0].borrow_mut().grad += b * x.grad;
        x.children[1].borrow_mut().grad += a * x.grad;
    });

    out
});

impl_op_ex!(*|a: &Variable, b: f64| -> Variable { a * Variable::from(b) });
impl_op_ex!(*|a: f64, b: &Variable| -> Variable { b * Variable::from(a) });

impl_op_ex!(/|a: &Variable, b: &Variable| -> Variable {
    assert!(a.data() != 0.0, "dividing by zero"); // todo: refactor
    let out = Variable::from(a.borrow().data / b.borrow().data);
    out.borrow_mut().op = Some(Operation::DIV);
    out.borrow_mut().children = vec![Variable(Rc::clone(a)), Variable(Rc::clone(b))];
    out.borrow_mut().fun = Some(|x: &VariableData| {
        let a = x.children[0].borrow().data;
        let b = x.children[1].borrow().data;
        // todo: division by zero
        x.children[0].borrow_mut().grad += x.grad / b;
        x.children[1].borrow_mut().grad += a * -x.grad / (b.powf(2.));
    });

    out
});

impl_op_ex!(-|a: &Variable, b: &Variable| -> Variable { a + -b });
impl_op!(-|a: &Variable| -> Variable { a * -1.0 });

impl Variable {
    pub fn new(tensor: VariableData) -> Variable {
        Variable(Rc::new(RefCell::new(tensor)))
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
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

    fn dfs(&self, top_sort: &mut Vec<Variable>, used: &mut HashSet<Variable>) {
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
        let out = Variable::from(max(self.borrow().data, 0.0));
        out.borrow_mut().op = Some(Operation::Custom(String::from("relu")));
        out.borrow_mut().children = vec![self.clone()];
        out.borrow_mut().fun = Some(|x: &VariableData| {
            if x.children[0].borrow().data > 0.0 {
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
            let val = unsafe { expf64(x.children[0].borrow_mut().data) };
            x.children[0].borrow_mut().grad += x.grad * val;
        });
        out
    }

    pub fn silu(&self) -> Variable {
        self * &self.sigmoid()
    }

    pub fn sigmoid(&self) -> Variable {
        (&Variable::from(1.0) + &(-self).exp()).pow(-1.0)
    }

    pub fn zero_grad(&self) {
        // todo: why do we need it?
        assert!(self.borrow().children.is_empty());
        self.borrow_mut().grad = 0.0;
    }

    pub fn step(&self, lr: f64) {
        let grad = self.grad();
        self.borrow_mut().data -= lr * grad;
    }
}
