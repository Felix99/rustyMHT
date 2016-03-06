#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Sensor;
use tracking_sim::Dynamics;
use tracking_sim::Config;
use tracking_sim::Track;
use tracking_sim::Measurement;
use tracking_sim::Hypothesis;

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
        assert!(msr.data.dim(0) == self.msr_matrix.dim(0));
        assert!(hypo.state.dim(0) == self.msr_matrix.dim(1));

        let innovation = &msr.data - &self.msr_matrix.dot(&hypo.state);
        let innovation_covar = self.msr_matrix.dot(&hypo.covar).dot(&self.msr_matrix.transpose())
        + &self.msr_covar;
        let innovation_covar_inv = self.la.inv(&innovation_covar);
        let kalman_gain = hypo.covar.dot(&self.msr_matrix.transpose()).dot(&innovation_covar_inv);
        hypo.state = &hypo.state + &kalman_gain.dot(&innovation);
        hypo.covar = &hypo.covar - &kalman_gain.dot(&innovation_covar).dot(&kalman_gain.transpose());
        hypo.weight = if self.config.rho_F > 0.0 {
            self.config.p_D / self.config.rho_F * hypo.weight *
                self.la.normal(&innovation,&Tensor::<f64>::new(vec![0.0,0.0]).reshape(&[2,1]), &innovation_covar)
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
        let innovation_covar = self.msr_matrix.dot(&track.covar).dot(&self.msr_matrix.transpose())
        + &self.msr_covar;
        let innovation_covar_inv = self.la.inv(&innovation_covar);
        let z_hat = self.msr_matrix.dot(&track.state);
        let mut msrs_gated = vec![];
        for z in msrs.iter() {
            let nu = &z.data - &z_hat;
            let dist = nu.transpose().dot(&innovation_covar_inv).dot(&nu);
            if self.la.get(&dist,0,0) < self.config.mu_gating {
                let z_copy = Measurement::new(z.data.data().clone());
                msrs_gated.push(z_copy);
            }
        }
        msrs_gated
    }

}
