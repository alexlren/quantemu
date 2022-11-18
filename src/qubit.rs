use std::cmp::PartialEq;
use std::fmt::{Debug, Formatter, Error};

use nalgebra::{Complex, Vector2, UnitVector2};
use num_traits::identities::{One, Zero};
use rand::Rng;

#[derive(PartialEq)]
pub struct Qubit {
    pub data: UnitVector2<Complex<f64>>,
}

impl Qubit {
    pub fn zero() -> Self {
        Self {
            data: UnitVector2::new_unchecked(
                Vector2::new(Complex::one(), Complex::zero()),
            ),
        }
    }

    pub fn one() -> Self {
        Self {
            data: UnitVector2::new_unchecked(
                Vector2::new(Complex::zero(), Complex::one()),
            ),
        }
    }

    pub fn from_re(x: f64, y: f64) -> Self {
        Self {
            data: UnitVector2::new_normalize(
                Vector2::new(Complex::from(x), Complex::from(y))
            )
        }
    }

    pub fn from_complex(x: Complex<f64>, y: Complex<f64>) -> Self {
        Self {
            data: UnitVector2::new_normalize(Vector2::new(x, y))
        }
    }

    pub fn measure(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        let state: f64 = rng.gen();
        let z_prob = self.data.x.norm().powi(2);
        // We force the value to either (1, 0) or (0, 1) so it's normalized
        let data = self.data.as_mut_unchecked();

        if state >= z_prob {
            data.x = Complex::zero();
            data.y = Complex::one();
            true
        } else {
            data.x = Complex::one();
            data.y = Complex::zero();
            false
        }
    }
}

impl Debug for Qubit {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Qubit [{} {}]", self.data.x, self.data.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_zero() {
        let q0 = Qubit::zero();

        assert_eq!(q0.data.x.re, 1.0); assert_eq!(q0.data.x.im, 0.0);
        assert_eq!(q0.data.y.re, 0.0); assert_eq!(q0.data.y.im, 0.0);
    }

    #[test]
    fn test_qubit_one() {
        let q1 = Qubit::one();

        assert_eq!(q1.data.x.re, 0.0);
        assert_eq!(q1.data.x.im, 0.0);
        assert_eq!(q1.data.y.re, 1.0);
        assert_eq!(q1.data.y.im, 0.0);
    }

    #[test]
    fn test_qubit_init_from_re() {
        let q0 = Qubit::from_re(1.0, 0.0);
        let q1 = Qubit::from_re(0.0, 1.0);

        assert_eq!(q0, Qubit::zero());
        assert_eq!(q1, Qubit::one());
    }

    #[test]
    fn test_qubit_init_from_complex() {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        let q0 = Qubit::from_complex(one, zero);
        let q1 = Qubit::from_complex(zero, one);

        assert_eq!(q0, Qubit::zero());
        assert_eq!(q1, Qubit::one());
    }

    #[test]
    fn test_measure_0() {
        let mut q = Qubit::zero();

        assert_eq!(q.measure() as u8, 0);
    }

    #[test]
    fn test_measure_1() {
        let mut q = Qubit::one();

        assert_eq!(q.measure() as u8, 1);
    }
}
