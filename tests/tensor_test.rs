use rand::random;
use rust_minigrad::{Operation, Variable, VariableData};

macro_rules! as_close {
    ($left:expr, $right:expr, $tol:expr) => {{
        let (left, right, tol) = (&$left, &$right, &$tol);
        if !((*left - *right).abs() < *tol) {
            panic!(
                "assertion failed: `(left ~ right)`\n  left: `{}`,\n right: `{}`\n  diff: `{}`",
                *left,
                *right,
                (*left - *right).abs()
            );
        }
    }};
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn construct() {
        let id_ = random();
        let data = VariableData {
            data: 2.0,
            grad: 0.0,
            id: id_,
            fun: None,
            op: Some(Operation::ADD),
            children: Vec::new(),
        };

        let var = Variable::new(data);
        assert_eq!(var.0.borrow().id, id_);
        let s = &var * 3.;
        let m = &var + 6.;
        assert_eq!(s.0.borrow().data, 6.);
        assert_eq!(m.0.borrow().data, 8.);
    }

    #[test]
    fn simple_grad_add() {
        let a = Variable::from(3.);
        let b = Variable::from(4.);
        let mut c = &a + &b; // why &mut ?
        c.backward();
        assert_eq!(a.borrow().grad, 1.0);
    }

    #[test]
    fn simple_grad_add_self() {
        let a = Variable::from(3.);
        let mut c = &a + &a;
        c.backward();
        assert_eq!(a.borrow().grad, 2.0);
    }

    #[test]
    fn simple_grad_mul() {
        let a = Variable::from(3.);
        let b = Variable::from(4.);
        let mut c = &a * &b;
        c.backward();
        assert_eq!(a.borrow().grad, 4.0);
        assert_eq!(b.borrow().grad, 3.0);
    }

    #[test]
    fn simple_grad_self_mul() {
        let a = Variable::from(3.);
        let mut c = &a * &a;
        c.backward();
        assert_eq!(a.borrow().grad, 6.0);
    }

    #[test]
    fn simple_sin_test() {
        let x = Variable::from(3.);

        let mut y = x.sin();
        y.backward();

        // sin'(3) ~= -0.9899924966
        as_close!(x.borrow().grad, -0.9899924966, 0.0001);
    }

    #[test]
    fn simple_cos_test() {
        let x = Variable::from(3.);

        let mut y = x.cos();
        y.backward();
        // cos'(3) ~= -0.14112000806
        as_close!(x.borrow().grad, -0.14112000806, 0.0001);
    }

    #[test]
    fn simple_relu() {
        let x = Variable::from(5.);
        let y = Variable::from(10.);
        let mut z = (&x + &y).relu();
        z.backward();
        as_close!(x.borrow().grad, 1.0, 0.0001);
    }

    #[test]
    fn adv_relu() {
        let x = Variable::from(5.);
        let y = Variable::from(10.);
        let mut z = (&x + &y).pow(2.).relu();
        // d/dx [relu((x + y) ^ 2)] = 2x + 2y = 30
        z.backward();
        as_close!(x.borrow().grad, 30., 0.0001);
    }

    #[test]
    fn adv_relu_neg() {
        let x = Variable::from(-5.);
        let mut z = x.relu();
        z.backward();
        as_close!(x.borrow().grad, 0., 0.0001);
    }

    #[test]
    fn simple_exp() {
        let x = Variable::from(5);
        let mut y = x.exp();
        y.backward();
        as_close!(x.borrow().grad, y.borrow().data, 0.0001);
    }

    #[test]
    fn simple_sigmoid_test() {
        let x = Variable::from(2.0);
        let mut y = x.sigmoid();
        y.backward();
        as_close!(x.borrow().grad, 0.1049935854, 0.001);
    }
}
