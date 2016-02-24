#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Dynamics;

pub struct Target {
	pub state : Tensor<f64>,
	la : Linalg,
	dynamics: Dynamics
}

impl Target {
	pub fn new(state: Tensor<f64>, deltaT: f64, q: f64) -> Target {
		Target {
			state: state,
			la : Linalg::new(),
			dynamics : Dynamics::new(deltaT,q)
		}
	}

	pub fn move_forward(&mut self) {
		self.state = self.dynamics.F.dot(&self.state);
		let dim = self.state.dim(0);
		if self.dynamics.q > 0.0 {
			self.state = &self.state + &self.la.mvnrnd(&Tensor::<f64>::zeros(&[dim]), &self.dynamics.Q);
		}
	}
}
