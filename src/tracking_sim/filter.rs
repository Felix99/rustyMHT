#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Sensor;
use tracking_sim::Dynamics;
use tracking_sim::Config;

pub struct Filter {
    pub config : Config,
    pub msr_covar : Tensor<f64>,
    pub msr_matrix : Tensor<f64>,
    la : Linalg,
    dynamics : Dynamics
}

impl Filter {
    pub fn new() -> Filter {
        let config = Config::new();
        let la = Linalg::new();
        Filter {
            config : Config::new(),
            msr_covar : la.copy(&config.msr_covar),
            msr_matrix : la.copy(&config.msr_matrix),
            la : Linalg::new(),
            dynamics : Dynamics::new(config.delta_t, config.q),
        }
    }
}
