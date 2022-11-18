use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Complex, DVector, Unit};
use num_traits::identities::{One, Zero};

use crate::qubit::Qubit;

#[derive(Debug, PartialEq)]
pub struct Register {
    pub data: Unit<DVector<Complex<f64>>>,
}

impl Register {
    /// Build a register from a slice of complex values
    pub fn from_slice(values: &[Complex<f64>]) -> Self {
        Self {
            data: Unit::new_normalize(DVector::from_row_slice(values))
        }
    }

    /// Initialize a Register from a number of qubit n and a positive integer i
    /// Register::from_int(3, 4); // |100>
    pub fn from_int(n: u32, i: u32) -> Self {
        let vlen = 2u32.pow(n);
        let mut data = DVector::<Complex<f64>>::zeros(vlen as usize);
        data[i as usize] = Complex::one();

        Self {
            data: Unit::new_normalize(data),
        }
    }

    /// Tensor product of 2 registers into a single register
    pub fn tensor_product(&self, other: &Self) -> Self {
        Self {
            data: Unit::new_normalize(self.data.kronecker(&other.data))
        }
    }
}

impl From<Qubit> for Register {
    /// Transform a qubit into a register
    fn from(q: Qubit) -> Self {
        let data = q.data.into_inner();

        Self::from_slice(&[data.x, data.y])
    }
}

impl AbsDiffEq for Register {
    type Epsilon = <DVector<Complex<f64>> as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        <DVector<Complex<f64>> as AbsDiffEq>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.data.abs_diff_eq(&other.data, epsilon)
    }
}

impl RelativeEq for Register {
    fn default_max_relative() -> Self::Epsilon {
        <DVector<Complex<f64>> as RelativeEq>::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.data.relative_eq(&other.data, epsilon, max_relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_init_qubit() {
        let r: Register = Qubit::zero().into();

        assert_eq!(r.data[0], Complex::one());
        assert_eq!(r.data[1], Complex::zero());
    }

    #[test]
    fn test_register_init_int() {
        let r: Register = Register::from_int(3, 4);

        assert_eq!(r.data[0], Complex::zero());
        assert_eq!(r.data[1], Complex::zero());
        assert_eq!(r.data[2], Complex::zero());
        assert_eq!(r.data[3], Complex::zero());
        assert_eq!(r.data[4], Complex::one());
        assert_eq!(r.data[5], Complex::zero());
        assert_eq!(r.data[6], Complex::zero());
        assert_eq!(r.data[7], Complex::zero());
    }

    #[test]
    fn test_register_tensor_product() {
        let b1: Register = Qubit::one().into();
        let b0: Register = Qubit::zero().into();

        let r = b1.tensor_product(&b0);
        assert_eq!(r.data[0], Complex::zero());
        assert_eq!(r.data[1], Complex::zero());
        assert_eq!(r.data[2], Complex::one());
        assert_eq!(r.data[3], Complex::zero());
        assert_eq!(r, Register::from_int(2, 2));

        let r2 = b1.tensor_product(&b0).tensor_product(&b0);
        assert_eq!(r2, Register::from_int(3, 4));
    }
}
