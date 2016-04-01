#![feature(box_syntax, box_patterns)]

use rm::linalg::matrix::Matrix;
use tracking_sim::Target;
use tracking_sim::Linalg;

pub struct Dynamics {
	pub F : Matrix<f64>,
	pub Q : Matrix<f64>,
	pub q : f64
}

impl Dynamics {

	pub fn new(deltaT: f64, q: f64,) -> Dynamics {
		Dynamics {
			F : Matrix::new(4,4,vec![1.0, 0.0, deltaT, 0.0,
			0.0, 1.0, 0.0, deltaT,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0]),
			
			Q : Matrix::new(4,4,vec![0.3 * deltaT.powf(3.0), 0.0, 0.5 * deltaT.powf(2.0), 0.0,
			0.0, 0.3 * deltaT.powf(3.0), 0.0, 0.5 * deltaT.powf(2.0),
			0.5 * deltaT.powf(2.0), 0.0, deltaT, 0.0,
			0.0, 0.5 * deltaT.powf(2.0), 0.0, deltaT]) * q,
			
			q : q
		}
	}

}
