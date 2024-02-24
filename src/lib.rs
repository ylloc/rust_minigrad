#![feature(core_intrinsics)]

mod variable;
pub use crate::variable::{Operation, Variable, VariableData};

mod tensor;
pub use crate::tensor::{Tensor1D, Tensor2D};

mod utils;
pub use crate::utils::max;
