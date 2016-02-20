#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;

pub struct Target {
	pub state : Tensor<f64>
}

impl Target {
	pub fn new(state: Tensor<f64>) -> Target {
		Target {
			state: state
		}
	}
}
