use std::f64::consts::FRAC_1_SQRT_2;
use std::ops::Deref;
use nalgebra::{Complex, DMatrix, DVector, Unit};
use num_traits::identities::{One, Zero};

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

    pub fn not() -> Self {
        // |0  1|
        // |1  0|
        Self::from(MatrixOperator::new_normalize(
            DMatrix::from_row_slice(2, 2, &[
                Complex::zero(), Complex::one(),
                Complex::one(), Complex::zero(),
            ]))
        )
    }

    fn apply_and_get(&self, r: &Register) -> Unit<DVector<Complex<f64>>> {
        Unit::new_normalize(self.matrix.deref() * r.data.deref())
    }

    pub fn apply_mut(&self, r: &mut Register) {
        r.data = self.apply_and_get(r);
    }

    pub fn apply(&self, r: &Register) -> Register {
        let r = self.apply_and_get(r);
        let values = r.into_inner();

        Register::from_slice(values.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::qubit::Qubit;

    #[test]
    pub fn test_hadamard_0() {
        let mut q = Qubit::zero().into();
        let h = Gate::hadamard();

        h.apply_mut(&mut q);
        assert_relative_eq!(q, Qubit::from_re(FRAC_1_SQRT_2, FRAC_1_SQRT_2).into());
        h.apply_mut(&mut q);
        assert_relative_eq!(q, Qubit::zero().into());
    }

    #[test]
    pub fn test_hadamard_1() {
        let mut q = Qubit::one().into();
        let h = Gate::hadamard();

        h.apply_mut(&mut q);
        assert_relative_eq!(q, Qubit::from_re(FRAC_1_SQRT_2, -FRAC_1_SQRT_2).into());
        h.apply_mut(&mut q);
        assert_relative_eq!(q, Qubit::one().into());
    }

    #[test]
    pub fn test_not() {
        let mut q = Qubit::zero().into();
        let x = Gate::not();

        x.apply_mut(&mut q);
        assert_eq!(q, Qubit::one().into());
        x.apply_mut(&mut q);
        assert_eq!(q, Qubit::zero().into());
    }
}
