#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::linalg::Linalg;
use tracking_sim::target::Target;

pub struct Sensor {
    pub msr_covar : Tensor<f64>,
    pub msr_matrix : Tensor<f64>,
    la : Linalg,
}

impl Sensor {
    pub fn new(msr_covar: Tensor<f64>, msr_matrix: Tensor<f64>) -> Sensor {
        Sensor {
            msr_covar : msr_covar,
            msr_matrix : msr_matrix,
            la : Linalg::new(),
        }
    }

    pub fn measure(&mut self, target: &Target) -> Tensor<f64> {
        self.la.mvnrnd(&(self.msr_matrix.dot(&target.state)), &self.msr_covar)
    }
}
