use rm::prelude::*;
use tracking_sim::IsSimilar;



pub struct Hypothesis {
    pub state : Matrix<f64>,
    pub covar : Matrix<f64>,
    pub weight : f64,
}

impl Hypothesis {
    pub fn new(state : &Matrix<f64>, covar: &Matrix<f64>, weight : f64) -> Hypothesis {
        Hypothesis {
            state : state.clone(),
            covar : covar.clone(),
            weight : weight,
        }
    }

}

impl Clone for Hypothesis {
    fn clone(&self) -> Hypothesis {
        Hypothesis::new(&self.state,&self.covar,self.weight)
    }
}

impl IsSimilar for Hypothesis {
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