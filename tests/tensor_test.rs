use rand::random;
use rust_minigrad::{Operation, Variable, VariableData};

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
        let mut c = &a * &Variable::from(2.);
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
                loss = &loss + &loss_fn(&(&(&a * &Variable::from(x)) + &b), &Variable::from(y))
            }

            loss = &loss / &Variable::from(lin.len() as f64);

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
        println!("{}", cnt);
    }

    #[test]
    fn rc_check_s() {
        let x = Variable::from(3.);
        let mut a;
        {
            a = &x * &Variable::from(0.3);
        }
        a.backward();
        println!("{}", x.grad());
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
}
