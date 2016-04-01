#[macro_use(Matrix)]
use rm::linalg::matrix::Matrix;
use tracking_sim::Linalg;
use tracking_sim::Sensor;
use tracking_sim::Dynamics;
use tracking_sim::Config;
use tracking_sim::Track;
use tracking_sim::Measurement;
use tracking_sim::Hypothesis;

pub struct Filter {
    pub config : Config,
    pub msr_covar : Matrix<f64>,
    pub msr_matrix : Matrix<f64>,
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
                h.state = &self.dynamics.F * &h.state;
                h.covar = &self.dynamics.F * &h.covar * &self.dynamics.F.transpose() + &self.dynamics.Q;
            }
        }
        track.update_state()
    }

    pub fn update(&self, track: &mut Track, msr: &Measurement) {
        assert!(msr.data.rows() == self.msr_matrix.rows());
        assert!(track.state.rows() == self.msr_matrix.cols());
        if track.hypotheses.is_empty() {
            let mut h = Hypothesis::new(&track.state,&track.covar,1.0);
            self.update_hypothesis(&mut h, msr);
            track.hypotheses = vec![h];
            track.update_state();
        } else {
            unsafe {
                for h in track.hypotheses.as_mut_slice() {
                    self.update_hypothesis(h, msr);
                }
            }
            track.update_state();
        }

    }

    pub fn update_hypothesis(&self, hypo: &mut Hypothesis, msr: &Measurement) {
        assert!(msr.data.rows() == self.msr_matrix.rows());
        assert!(hypo.state.rows() == self.msr_matrix.cols());

        let innovation = &msr.data - &self.msr_matrix * &hypo.state;
        let innovation_covar = &self.msr_matrix * &hypo.covar * &self.msr_matrix.transpose()
        + &self.msr_covar;
        let innovation_covar_inv = innovation_covar.inverse();
        let kalman_gain = &hypo.covar * &self.msr_matrix.transpose() * &innovation_covar_inv;
        hypo.state = &hypo.state + &kalman_gain * &innovation;
        hypo.covar = &hypo.covar - &kalman_gain * &innovation_covar * &kalman_gain.transpose();
        hypo.weight = if self.config.rho_F > 0.0 {
            self.config.p_D / self.config.rho_F * hypo.weight *
                self.la.normal(&innovation,&Matrix::<f64>::new(2,1,vec![0.0,0.0]), &innovation_covar)
        } else {
            1_f64
        };
    }

    pub fn update_mht(&self, track: &mut Track, msr: &Vec<Measurement>) {
        let mut new_hypos = Vec::new();
        let msrs_gated = self.gate(track,msr);
        for h in &track.hypotheses {
            new_hypos.extend(self.create_hypotheses(&h,&msrs_gated));
        }
        track.hypotheses = new_hypos;
        track.update_state();
    }

    pub fn create_hypotheses(&self, hypo: &Hypothesis, msrs: &Vec<Measurement>) -> Vec<Hypothesis> {
        let state = self.la.copy(&hypo.state);
        let covar = self.la.copy(&hypo.covar);
        let p_0 = (1_f64-self.config.p_D) * hypo.weight;
        let h0 = Hypothesis::new(&state, &covar, p_0);
        let mut hypos = vec![h0];
        for z in msrs.iter() {
            let mut h_i = Hypothesis::new(&state, &covar, hypo.weight);
            self.update_hypothesis(&mut h_i,z);
            hypos.push(h_i);
        }
        hypos
    }

    pub fn gate(&self, track: &Track, msrs: &Vec<Measurement>) -> Vec<Measurement> {
        let innovation_covar = &self.msr_matrix * &track.covar * &self.msr_matrix.transpose()
        + &self.msr_covar;
        let innovation_covar_inv = innovation_covar.inverse();
        let z_hat = &self.msr_matrix * &track.state;
        let mut msrs_gated = vec![];
        for z in msrs.iter() {
            let nu = &z.data - &z_hat;
            let dist = (nu.transpose() * &innovation_covar_inv * &nu)[[0,0]];
            if dist < self.config.mu_gating {
                let z_copy = z.copy();
                msrs_gated.push(z_copy);
            }
        }
        msrs_gated
    }

}
