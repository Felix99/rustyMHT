#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::target::Target;

pub struct Dynamics {
	pub F : Tensor<f64>,
	pub Q : Tensor<f64>,
	q : f64
}

impl Dynamics {
	pub fn new(q: f64, T: f64) -> Dynamics {
		Dynamics {
			F : tensor![1.0, 0.0, T, 0.0; 
			0.0, 1.0, 0.0, T; 
			0.0, 0.0, 1.0, 0.0; 
			0.0, 0.0, 0.0, 1.0],
			
			Q : tensor![0.3 * T.powf(3.0), 0.0, 0.5 * T.powf(2.0), 0.0;
			0.0, 0.3 * T.powf(3.0), 0.0, 0.5 * T.powf(2.0);
			0.5 * T.powf(2.0), 0.0, T, 0.0;
			0.0, 0.5 * T.powf(2.0), 0.0, T] * q,
			
			q : q
			
		}
	}
	
	pub fn move_target(&self, target: &mut Target) {
		target.state = self.F.dot(&target.state);
		let mut rs = RandomState::new(1234);
		let stn = rs.normal::<f64>(&[4]);
		println!("Stn : {}",stn);
	}
}
