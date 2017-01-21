use rm::prelude::*;
use tracking_sim::Linalg;
use tracking_sim::Dynamics;

pub struct Target {
	pub state : Matrix<f64>,
	la : Linalg,
	dynamics: Dynamics
}

impl Target {
	pub fn new(state: Matrix<f64>, delta_t: f64, q: f64) -> Target {
		Target {
			state: state,
			la : Linalg::new(),
			dynamics : Dynamics::new(delta_t,q)
		}
	}

	pub fn move_forward(&mut self) {
		self.state = &self.dynamics.F * &self.state;
		let dim = self.state.rows();
		if self.dynamics.q > 0.0 {
			self.state = &self.state + &self.la.mvnrnd(&Matrix::<f64>::zeros(dim,1), &self.dynamics.Q);
		}
	}
}
