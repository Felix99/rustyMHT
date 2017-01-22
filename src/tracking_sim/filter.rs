use rm::prelude::*;
use tracking_sim::Linalg;
use tracking_sim::Dynamics;
use tracking_sim::Config;
use tracking_sim::Track;
use tracking_sim::Measurement;
use tracking_sim::Hypothesis;
use tracking_sim::Merger;

pub struct Filter {
    pub config : Config,
    pub msr_covar : Matrix<f64>,
    pub msr_matrix : Matrix<f64>,
    la : Linalg,
    dynamics : Dynamics
}

impl Filter {
    pub fn new(config: Config) -> Filter {
        Filter {
            config : config.clone(),
            msr_covar : config.msr_covar.clone(),
            msr_matrix : config.msr_matrix.clone(),
            la : Linalg::new(),
            dynamics : Dynamics::new(config.delta_t, config.q),
        }
    }

    pub fn set_config(&mut self, config: Config) {
        self.msr_covar = config.msr_covar.clone();
        self.msr_matrix = config.msr_matrix.clone();
        self.dynamics = Dynamics::new(config.delta_t, config.q);
        self.config = config;
    }

    pub fn predict(&self, track: &mut Track) {
        for i in 0..track.hypotheses.len() {
            let mut h = &mut track.hypotheses[i];
            h.state = &self.dynamics.F * &h.state;
            h.covar = &self.dynamics.F * &h.covar * &self.dynamics.F.transpose() + &self.dynamics.Q;
        }
        track.update_state()
    }

    pub fn update(&self, track: &mut Track, msr: &Measurement) {
        assert!(msr.data.rows() == self.msr_matrix.rows());
        assert!(track.state.rows() == self.msr_matrix.cols());
        //
        if track.hypotheses.is_empty() {
            let mut h = Hypothesis::new(&track.state,&track.covar,1.0);
            self.update_hypothesis(&mut h, msr);
            track.hypotheses = vec![h];
            track.update_state();
        } else {
            for h in track.hypotheses.as_mut_slice() {
                    self.update_hypothesis(h, msr);
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
        if innovation_covar_inv.is_ok() {
            let kalman_gain = &hypo.covar * &self.msr_matrix.transpose() * &innovation_covar_inv.unwrap();
            hypo.state = &hypo.state + &kalman_gain * &innovation;
            hypo.covar = &hypo.covar - &kalman_gain * &innovation_covar * &kalman_gain.transpose();
            hypo.weight = if self.config.rho_F > 0.0 {
                self.config.p_D / self.config.rho_F * hypo.weight *
                    self.la.normal(&innovation,&Matrix::<f64>::new(2,1,vec![0.0,0.0]), &innovation_covar)
            } else {
                1_f64
            };
        } else {
            panic!("Matrix inversion failed!")
        }

    }

    pub fn update_mht(&self, track: &mut Track, msr: &Vec<Measurement>) {
        let mut new_hypos = Vec::new();
        let msrs_gated = self.gate(track,msr);
        for h in &track.hypotheses {
            let (new_hypos_for_h,delta) = self.create_hypotheses(&h,&msrs_gated);
            track.lr *= delta;
            new_hypos.extend(new_hypos_for_h);
        }
        track.hypotheses = new_hypos;
        self.prune_hypotheses(track);
        track.hypotheses = self.merge(&track.hypotheses, 1_f64);
        track.update_state();
    }
    // returns: new hypotheses together with lr delta
    pub fn create_hypotheses(&self, hypo: &Hypothesis, msrs: &Vec<Measurement>) -> (Vec<Hypothesis>,f64) {
        let state = &hypo.state;
        let covar = &hypo.covar;
        let p_0 = (1_f64-self.config.p_D) * hypo.weight;
        let mut delta = p_0;
        let h0 = Hypothesis::new(&state, &covar, p_0);
        let mut hypos = vec![h0];
        for z in msrs.iter() {
            let mut h_i = Hypothesis::new(&state, &covar, hypo.weight);
            self.update_hypothesis(&mut h_i,z);
            delta += h_i.weight;
            hypos.push(h_i);
        }
        (hypos,delta)
    }

    pub fn gate(&self, track: &Track, msrs: &Vec<Measurement>) -> Vec<Measurement> {
        let innovation_covar = &self.msr_matrix * &track.covar * &self.msr_matrix.transpose()
        + &self.msr_covar;
        let innovation_covar_inv_res = innovation_covar.inverse();
        if innovation_covar_inv_res.is_ok() {
            let innovation_covar_inv = innovation_covar_inv_res.unwrap();
            let z_hat = &self.msr_matrix * &track.state;
            let mut msrs_gated = vec![];
            for z in msrs.iter() {
                let nu = &z.data - &z_hat;
                let dist = (nu.transpose() * &innovation_covar_inv * &nu)[[0,0]];
                if dist < self.config.mu_gating {
                    msrs_gated.push(z.clone());
                }
            }
            msrs_gated
        } else {
            panic!("Matrix inversion failed!")
        }
    }

    pub fn prune_hypotheses(&self, track: &mut Track) {
        track.hypotheses = track.hypotheses.iter().filter(
            |e| e.weight > self.config.threshold_pruning).cloned().collect::<Vec<_>>();
        track.normalize_weights();
    }
}

impl Merger<Hypothesis> for Filter {
    fn merge_all(&self, to_merge: Vec<Hypothesis>) -> Hypothesis {
        let dim = &to_merge[0].state.rows();
        let weight = to_merge.iter().fold(0_f64, |a,e| a + e.weight);
        let mean = to_merge.iter().fold(Matrix::<f64>::zeros(*dim,1), |a,e| {
            a + &e.state * e.weight
        });
        let covar = to_merge.iter().fold(Matrix::<f64>::zeros(*dim,*dim), |a,e| {
            let v = &e.state - &mean;
            let spread = &v * &v.transpose();
            let unweighted = &e.covar + spread;
            a + &unweighted * e.weight
        });
        Hypothesis::new(&mean,&covar,weight)
    }
}