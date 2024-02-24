#![feature(core_intrinsics)]

mod variable;
pub use crate::variable::{Operation, Variable, VariableData};

mod tensor;

mod utils;
pub use crate::utils::max;
