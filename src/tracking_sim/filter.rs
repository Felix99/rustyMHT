#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Sensor;
use tracking_sim::Dynamics;
use tracking_sim::Config;
use tracking_sim::Track;
use tracking_sim::Measurement;

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

    pub fn set_config(&mut self, config: Config) {
        self.msr_covar = self.la.copy(&config.msr_covar);
        self.msr_matrix = self.la.copy(&config.msr_matrix);
        self.dynamics = Dynamics::new(config.delta_t, config.q);
        self.config = config;
    }

    pub fn predict(&self, track: &mut Track) {
        unsafe {
            for h in track.hypotheses.as_mut_slice() {
                h.state = self.dynamics.F.dot(&h.state);
                h.covar = self.dynamics.F.dot(&h.covar).dot(&self.dynamics.F.transpose()) + &self.dynamics.Q;
            }
        }
        track.update_state()
    }

    pub fn update(&self, track: &mut Track, msr: &Measurement) {
        assert!(msr.data.dim(0) == self.msr_matrix.dim(0));
        assert!(track.state.dim(0) == self.msr_matrix.dim(1));

        let innovation = &msr.data - &self.msr_matrix.dot(&track.state);
        let innovation_covar = self.msr_matrix.dot(&track.covar).dot(&self.msr_matrix.transpose())
            + &self.msr_covar;
        let innovation_covar_inv = self.la.inv(&innovation_covar);
        let kalman_gain = track.covar.dot(&self.msr_matrix.transpose()).dot(&innovation_covar_inv);
        track.state = &track.state + &kalman_gain.dot(&innovation);
        track.covar = &track.covar - &kalman_gain.dot(&innovation_covar).dot(&kalman_gain.transpose());

    }

    pub fn update_mht(&self, track: &mut Track, msr: &Vec<Measurement>) {

    }

}
