use rand::random;
use rust_minigrad::{Operation, Tensor, TensorData};

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

    #[test]
    fn simple_sin_test() {
        let x = Tensor::from(3.);

        let mut y = x.sin();
        y.backward();

        // sin'(3) ~= -0.9899924966
        as_close!(x.borrow().grad, -0.9899924966, 0.0001);
    }

    #[test]
    fn simple_cos_test() {
        let x = Tensor::from(3.);

        let mut y = x.cos();
        y.backward();

        // cos'(3) ~= -0.14112000806
        as_close!(x.borrow().grad, -0.14112000806, 0.0001);
    }
}
