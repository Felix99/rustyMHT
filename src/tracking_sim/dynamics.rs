use rm::linalg::matrix::Matrix;

pub struct Dynamics {
	pub F : Matrix<f64>,
	pub Q : Matrix<f64>,
	pub q : f64,
}

impl Dynamics {

	pub fn new(delta_t: f64, q: f64,) -> Dynamics {
		Dynamics {
			F : Matrix::new(4,4,vec![1.0, 0.0, delta_t, 0.0,
			0.0, 1.0, 0.0, delta_t,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0]),

			Q : Matrix::new(4,4,vec![0.3 * delta_t.powf(3.0), 0.0, 0.5 * delta_t.powf(2.0), 0.0,
			0.0, 0.3 * delta_t.powf(3.0), 0.0, 0.5 * delta_t.powf(2.0),
			0.5 * delta_t.powf(2.0), 0.0, delta_t, 0.0,
			0.0, 0.5 * delta_t.powf(2.0), 0.0, delta_t]) * q,
			
			q : q
		}
	}

}
