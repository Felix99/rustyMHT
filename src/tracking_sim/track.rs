//use rm::linalg::Matrix;
use rm::prelude::*;

use tracking_sim::Hypothesis;
use tracking_sim::IsSimilar;

pub struct Track {
    pub state : Matrix<f64>,
    pub covar : Matrix<f64>,
    pub time : f64,
    pub hypotheses : Vec<Hypothesis>,
    dim : usize,
    pub id : i64,
    pub lr : f64,
}

impl Track {
    pub fn new(state : &Matrix<f64>, covar: &Matrix<f64>, time : f64) -> Track {
        Track {
            state : state.clone(),
            covar : covar.clone(),
            time : time,
            hypotheses : vec![Hypothesis::new(state,covar,1.0)],
            dim : state.rows(),
            id : -1,
            lr : 1_f64,
        }
    }

    pub fn update_state(&mut self) {
        assert!(!self.hypotheses.is_empty());
        self.normalize_weights();
        self.state = self.hypotheses.iter().fold(Matrix::zeros(self.dim,1), |s,e| {
            s + &e.state * e.weight
        });

        self.covar = self.hypotheses.iter().fold(Matrix::zeros(self.dim,self.dim), |a,e| {
            let v = &e.state - &self.state;
            let spread = &v * &v.transpose();
            let unweighted = &e.covar + spread;
            a + unweighted * e.weight
        });

    }

    pub fn normalize_weights(&mut self) {
        let sum = self.hypotheses.iter().fold(0_f64, |a,e| a + e.weight);

        for h in self.hypotheses.as_mut_slice() {
                h.weight = h.weight / sum;
        }

    }

    pub fn merge_hypotheses(&mut self) {
        let mut merged_hypos = Vec::new();
        let hypotheses_copy = self.hypotheses.clone();
        merged_hypos = self.merge_local(hypotheses_copy, merged_hypos);
        self.hypotheses = merged_hypos;
    }

    fn merge_local(&self, list_of_hypos: Vec<Hypothesis>, mut merged_hypos: Vec<Hypothesis>) -> Vec<Hypothesis> {
        if list_of_hypos.len() > 1 {
            let separated = self.find_elements_to_merge_with_tail(list_of_hypos);
            let to_merge = separated.0;
            let not_to_merge = separated.1;
            merged_hypos.push(self.merge_all(to_merge));
            self.merge_local(not_to_merge,merged_hypos)
        } else {
            merged_hypos
        }
    }

    fn find_elements_to_merge_with_tail(&self, mut list_of_hypos : Vec<Hypothesis>) -> (Vec<Hypothesis>,Vec<Hypothesis>) {
        let h_tail = list_of_hypos.pop().unwrap();
        let mut to_merge = vec![h_tail.clone()];
        let mut not_to_merge = Vec::new();
        for h in list_of_hypos.iter() {
            if self.hypotheses_are_close(&h,&h_tail) {
                to_merge.push(h.clone());
            } else {
                not_to_merge.push(h.clone());
            }
        }
        (to_merge,not_to_merge)
    }

    fn hypotheses_are_close(&self, h1: &Hypothesis, h2: &Hypothesis) -> bool {
        let d = &h1.state - &h2.state;
        let metric = (&h1.covar + &h2.covar).inverse();
        if metric.is_ok() {
            let stat_dist = &d.transpose() * metric.unwrap() * &d;
            stat_dist[[0,0]]< 12.23
        } else {
            panic!("Matrix inversion failed!")
        }

    }

    fn merge_all(&self, to_merge: Vec<Hypothesis>) -> Hypothesis {
        let weight = to_merge.iter().fold(0_f64, |a,e| a + e.weight);
        let mean = to_merge.iter().fold(Matrix::<f64>::zeros(self.dim,1), |a,e| {
            a + &e.state * e.weight
        });
        let covar = to_merge.iter().fold(Matrix::<f64>::zeros(self.dim,self.dim), |a,e| {
            let v = &e.state - &mean;
            let spread = &v * &v.transpose();
            let unweighted = &e.covar + spread;
            a + &unweighted * e.weight
        });
        Hypothesis::new(&mean,&covar,weight)
    }

    pub fn prune_hypotheses(&mut self, threshold : f64) {
        self.hypotheses = self.hypotheses.iter().filter(
            |e| e.weight > threshold).cloned().collect::<Vec<_>>();
    }

}

impl Clone for Track {
    fn clone(&self) -> Track {
        Track {
            state : self.state.clone(),
            covar : self.covar.clone(),
            time : self.time,
            hypotheses : self.hypotheses.clone(),
            dim : self.dim,
            lr : self.lr,
            id : self.id,
        }
    }
}

impl IsSimilar for Track {
    fn is_similar(&self, other: &Self, mu_similar: f64) -> bool {
        let d = &self.state - &other.state;
        let covar = &self.covar + &other.covar;
        let covar_inv_res = covar.inverse();
        if covar_inv_res.is_ok() {
            let covar_inv = covar_inv_res.unwrap();
            let stat_dist = (d.transpose() * &covar_inv * &d)[[0, 0]];
            return stat_dist < mu_similar;
        } else {
            return false;
        }

    }

}

