use rand::random;
use rust_minigrad::{Operation, Tensor1D, Tensor2D, Variable, VariableData};
macro_rules! assert_close {
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
    use std::pin::Pin;

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
        assert_eq!(s.data(), 6.);
        assert_eq!(m.data(), 8.);
    }

    #[test]
    fn simple_grad_add() {
        let a = Variable::from(3.);
        let b = Variable::from(4.);
        let mut c = &a + &b; // why &mut ?
        c.backward();
        assert_eq!(a.grad(), 1.0);
    }

    #[test]
    fn simple_grad_add_self() {
        let a = Variable::from(3.);
        let mut c = &a + &a;
        c.backward();
        assert_eq!(a.grad(), 2.0);
    }

    #[test]
    fn simple_grad_mul() {
        let a = Variable::from(3.);
        let b = Variable::from(4.);
        let mut c = &a * &b;
        c.backward();
        assert_eq!(a.grad(), 4.0);
        assert_eq!(b.grad(), 3.0);
    }

    #[test]
    fn simple_grad_self_mul() {
        let a = Variable::from(3.);
        let mut c = &a * &a;
        c.backward();
        assert_eq!(a.grad(), 6.0);
    }

    #[test]
    fn mul_const() {
        let a = Variable::from(3.);
        let mut c = &a * Variable::from(2.);
        c.backward();
        assert_eq!(a.grad(), 2.);
    }

    #[test]
    fn simple_sin_test() {
        let x = Variable::from(3.);

        let mut y = x.sin();
        y.backward();

        // sin'(3) ~= -0.9899924966
        assert_close!(x.grad(), -0.9899924966, 0.0001);
    }

    #[test]
    fn simple_cos_test() {
        let x = Variable::from(3.);

        let mut y = x.cos();
        y.backward();
        // cos'(3) ~= -0.14112000806
        assert_close!(x.grad(), -0.14112000806, 0.0001);
    }

    #[test]
    fn simple_relu() {
        let x = Variable::from(5.);
        let y = Variable::from(10.);
        let mut z = (&x + &y).relu();
        z.backward();
        assert_close!(x.grad(), 1.0, 0.0001);
    }

    #[test]
    fn adv_relu() {
        let x = Variable::from(5.);
        let y = Variable::from(10.);
        let mut z = (&x + &y).pow(2.).relu();
        // println!("{:?}", z);
        // d/dx [relu((x + y) ^ 2)] = 2x + 2y = 30
        z.backward();
        assert_close!(x.grad(), 30., 0.0001);
    }

    #[test]
    fn adv_relu_neg() {
        let x = Variable::from(-5.);
        let mut z = x.relu();
        z.backward();
        assert_close!(x.grad(), 0., 0.0001);
    }

    #[test]
    fn simple_exp() {
        let x = Variable::from(5);
        let mut y = x.exp();
        y.backward();
        assert_close!(x.grad(), y.data(), 0.0001);
    }

    #[test]
    fn simple_sigmoid_test() {
        let x = Variable::from(2.0);
        let mut y = x.sigmoid();
        y.backward();
        assert_close!(x.grad(), 0.1049935854, 0.001);
    }

    #[test]
    fn simple_div() {
        let x = Variable::from(2.0);
        let y = &x * &x;
        let mut z = &Variable::from(4.0) / &y;
        z.backward();
        assert_close!(x.grad(), -1., 0.001);
    }

    #[test]
    fn hard_test() {
        let x = Variable::from(2.0);
        let y = &x * &x;
        let mut z = &(&y * &y) + &x;
        z.backward();
        assert_close!(x.grad(), 33.0, 0.0001);
    }

    #[test]
    fn linregression() {
        fn loss_fn(x: &Variable, y: &Variable) -> Variable {
            (x - y).pow(2.)
        }
        let a = Variable::from(0.1);
        let b = Variable::from(0.1);

        let (k, l) = (23.1, 16.77);
        let lin = (0..10)
            .map(|_| random::<f64>())
            .map(|x| (x, k * x + l))
            .collect::<Vec<_>>();

        let mut loss_f64 = 1.0;

        let mut cnt = 1;

        while loss_f64 >= 0.0001 {
            let mut loss = Variable::from(0.0);
            for &(x, y) in &lin {
                loss = &loss + &loss_fn(&((&a * Variable::from(x)) + &b), &Variable::from(y))
            }

            loss = &loss / Variable::from(lin.len() as f64);

            loss.backward();
            loss_f64 = loss.data();

            a.step(0.3);
            b.step(0.3);
            a.zero_grad();
            b.zero_grad();

            cnt += 1;
        }

        assert_close!(a.data(), k, 0.1);
        assert_close!(b.data(), l, 0.1);
    }

    #[test]
    fn rc_check_s() {
        let x = Variable::from(3.);
        let mut a;
        {
            a = &x * &Variable::from(0.3);
        }
        a.backward();
    }

    #[test]
    fn rc_check_h() {
        let x = Variable::from(1.);
        let y = x.clone();

        let x = &x * &x;
        let mut x = &x * &x;
        x.backward();
        assert_close!(y.grad(), 4., 0.1);
    }

    #[test]
    fn scope_test() {
        let mut x = Variable::from(0.1);
        let y = x.clone();
        {
            x = &x * &Variable::from(0.3);
            x = x.silu();
        }
        assert_eq!(x.borrow().children.len(), 2);
    }

    #[test]
    fn test_1_d() {
        let x = Tensor1D::new(2);
        *x.0.borrow_mut() = vec![Variable::from(1.0), Variable::from(2.0)];
        let y = &x.t() * &x;
        y.backward();
        assert_close!(x.borrow()[0].grad(), 2.0, 0.001);
        assert_close!(x.borrow()[1].grad(), 4.0, 0.001);
    }

    #[test]
    fn shape_test() {
        let x = Tensor2D::new(10, 5);
        let y = Tensor2D::new(5, 17);
        let z = &x * &y;
        assert_eq!(z.1, (10, 17));
    }

    #[test]
    fn test2() {
        let a = Tensor1D::new(6);
        *a.borrow_mut() = vec![
            Variable::from(1.),
            Variable::from(2.),
            Variable::from(3.),
            Variable::from(4.),
            Variable::from(5.),
            Variable::from(6.),
        ];
        let b = a.exp();
        assert_close!(b.borrow()[0].data(), 2.718281828459045, 0.001);
    }

    #[test]
    fn test3() {
        let a = Tensor1D::new(6);
        let b = a.clone();
        *a.borrow_mut() = vec![
            Variable::from(1.),
            Variable::from(2.),
            Variable::from(3.),
            Variable::from(4.),
            Variable::from(5.),
            Variable::from(6.),
        ];
        let x = a.t() * a;
        assert_close!(x.borrow()[0].data(), 91.0, 0.001);
        x.backward();
        for i in 0..6 {
            assert_close!(2. * (i + 1) as f64, b.borrow()[i].grad(), 0.001);
        }
    }

    #[test]
    fn test4() {
        let x = Tensor1D::from(&vec![1., 2., 3., 4., 5., 7.]);
        let z = x.clone();
        let y = &x.t() * x;
        y.backward();
        assert_close!(z.borrow_mut()[5].grad(), 14., 0.001);
    }

    #[test]
    fn ew() {
        let x = Tensor2D::from(&vec![vec![1., 2.], vec![1., 2.]]);
        let z = Variable::from(2.) * x;
    }

    #[test]
    fn f5() {
        let mut x = Tensor2D::from(&vec![vec![1., 2.], vec![1., 2.]]);
        x = x / Variable::from(0.2);
        assert_close!(x.borrow()[0][1].data(), 10.0, 0.0001);
    }
    #[test]
    fn f6() {
        let x = Tensor1D::from(&vec![1., 2., 3., 4., 5.]);
        let y = x.softmax();
        assert_close!(y.borrow()[0].data(), 0.011656230956, 0.001);
    }
}
