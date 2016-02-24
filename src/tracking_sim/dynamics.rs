#![feature(box_syntax, box_patterns)]
#[macro_use(tensor)]

use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Target;
use tracking_sim::Linalg;

pub struct Dynamics {
	pub F : Tensor<f64>,
	pub Q : Tensor<f64>,
	pub q : f64
}

impl Dynamics {

	pub fn new(deltaT: f64, q: f64,) -> Dynamics {
		Dynamics {
			F : tensor![1.0, 0.0, deltaT, 0.0;
			0.0, 1.0, 0.0, deltaT;
			0.0, 0.0, 1.0, 0.0; 
			0.0, 0.0, 0.0, 1.0],
			
			Q : tensor![0.3 * deltaT.powf(3.0), 0.0, 0.5 * deltaT.powf(2.0), 0.0;
			0.0, 0.3 * deltaT.powf(3.0), 0.0, 0.5 * deltaT.powf(2.0);
			0.5 * deltaT.powf(2.0), 0.0, deltaT, 0.0;
			0.0, 0.5 * deltaT.powf(2.0), 0.0, deltaT] * q,
			
			q : q
		}
	}

}
