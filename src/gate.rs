use nalgebra::{Complex, Matrix2, UnitVector2};

use crate::qubit::Qubit;

pub type GateMatrix = Matrix2<Complex<f64>>;

pub struct Gate {
    matrix: GateMatrix,
}

impl Gate {
    pub fn from(matrix: GateMatrix) -> Self {
        Self { matrix }
    }

    pub fn hadamard() -> Self {
        let coeff = 0.5_f64.sqrt();

        // 1/sqrt(2) . |1  1|
        //             |1 -1|
        Self::from(GateMatrix::new(
            Complex::from(coeff), Complex::from(coeff),
            Complex::from(coeff), Complex::from(-coeff),
        ))
    }

    pub fn apply(&self, q: &mut Qubit) {
        q.data = UnitVector2::new_normalize(self.matrix * q.data.into_inner());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_hadamard_0() {
        let mut q = Qubit::zero();
        let h = Gate::hadamard();
        let coeff = 0.5_f64.sqrt();

        h.apply(&mut q);
        assert_eq!(q, Qubit::from_re(coeff, coeff));
        h.apply(&mut q);
        assert_eq!(q, Qubit::zero());
    }

    #[test]
    pub fn test_hadamard_1() {
        let mut q = Qubit::one();
        let h = Gate::hadamard();
        let coeff = 0.5_f64.sqrt();

        h.apply(&mut q);
        assert_eq!(q, Qubit::from_re(coeff, -coeff));
        h.apply(&mut q);
        assert_eq!(q, Qubit::one());
    }
}
