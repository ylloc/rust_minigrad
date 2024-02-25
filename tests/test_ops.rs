use rand::random;
use rust_minigrad::{Operation, Tensor1D, Tensor2D, Variable, VariableData};

#[cfg(test)]
mod test {
    use rust_minigrad::{Tensor1D, Variable};

    #[test]
    fn test1() {
        let a = Variable::from(0.3);
        let x = a.clone();
        let mut a = 3.9 + &a + &a + a + 1.9;
        a.backward();
        assert_eq!(x.grad(), 3.0);
    }
}
