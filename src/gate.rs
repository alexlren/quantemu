use std::f64::consts::FRAC_1_SQRT_2;
use std::ops::Deref;
use nalgebra::{Complex, DMatrix, Unit};
use num_traits::identities::{One, Zero};

use crate::qubit::Qubit;
use crate::register::Register;

pub type MatrixOperator = Unit<DMatrix<Complex<f64>>>;

pub struct Gate {
    matrix: MatrixOperator,
}

impl Gate {
    pub fn from(matrix: MatrixOperator) -> Self {
        Self { matrix }
    }

    pub fn hadamard() -> Self {
        // 1/sqrt(2) . |1  1|
        //             |1 -1|
        Self::from(MatrixOperator::new_normalize(
            DMatrix::from_row_slice(2, 2, &[
                Complex::from(FRAC_1_SQRT_2), Complex::from(FRAC_1_SQRT_2),
                Complex::from(FRAC_1_SQRT_2), Complex::from(-FRAC_1_SQRT_2),
            ]))
        )
    }

    pub fn apply_mut(&self, r: &mut Register) {
        r.data = Unit::new_normalize(self.matrix.deref() * r.data.deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add support for approximation
    #[test]
    #[ignore]
    pub fn test_hadamard_0() {
        let mut q = Qubit::zero().into();
        let h = Gate::hadamard();

        h.apply_mut(&mut q);
        assert_eq!(q, Qubit::from_re(FRAC_1_SQRT_2, FRAC_1_SQRT_2).into());
        h.apply_mut(&mut q);
        assert_eq!(q, Qubit::zero().into());
    }

    #[test]
    #[ignore]
    pub fn test_hadamard_1() {
        let mut q = Qubit::one().into();
        let h = Gate::hadamard();

        h.apply_mut(&mut q);
        assert_eq!(q, Qubit::from_re(FRAC_1_SQRT_2, -FRAC_1_SQRT_2).into());
        h.apply_mut(&mut q);
        assert_eq!(q, Qubit::one().into());
    }
}
