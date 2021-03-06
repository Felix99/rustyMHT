use rm::linalg::Matrix;

pub struct Config {
    pub msr_covar: Matrix<f64>,
    pub msr_matrix: Matrix<f64>,
    pub q : f64,
    pub delta_t : f64,
    pub init_covar : Matrix<f64>,
    pub p_D : f64,
    pub rho_F : f64,
    pub mu_gating : f64,
    pub fov : Option<((f64,f64),(f64,f64))>,
    pub threshold_lr_upper : f64,
    pub threshold_lr_lower : f64,
    pub threshold_pruning : f64,
}

impl Config {
    pub fn new() -> Config {
        Config {
            msr_covar : Matrix::<f64>::identity(2),
            msr_matrix : Matrix::<f64>::new(2,4,vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            q : 1.0,
            delta_t : 1.0,
            init_covar :  Matrix::<f64>::identity(4) * 100.0,
            p_D : 0.95,
            rho_F : 1e-6,
            mu_gating : 9.21,
            fov : None,
            threshold_lr_upper : 1e3,
            threshold_lr_lower : 1e-3,
            threshold_pruning : 1e-2,
        }
    }
}

impl Clone for Config {
    fn clone(&self) -> Config {
        Config {
            msr_covar : self.msr_covar.clone(),
            msr_matrix : self.msr_matrix.clone(),
            q : self.q,
            delta_t : self.delta_t,
            init_covar :  self.init_covar.clone(),
            p_D : self.p_D,
            rho_F : self.rho_F,
            mu_gating : self.mu_gating,
            fov : self.fov,
            threshold_lr_upper : self.threshold_lr_upper,
            threshold_lr_lower : self.threshold_lr_lower,
            threshold_pruning : self.threshold_pruning,
        }
    }
}

pub struct SimConfig {
    pub steps : u32,
    pub num_targets : u32,
    pub fov : ((f64,f64),(f64,f64)),
    pub config : Config,
}

impl SimConfig {
    pub fn new() -> SimConfig {
        let config = Config {
            msr_covar : Matrix::<f64>::identity(2),
            msr_matrix : Matrix::<f64>::new(2,4,vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            q : 1.0,
            delta_t : 1.0,
            init_covar :  Matrix::<f64>::identity(4) * 100.0,
            p_D : 0.95,
            rho_F : 1e-6,
            mu_gating : 9.21,
            fov : Some(((0.0,1e3),(0.0,1e3))),
            threshold_lr_upper : 1e3,
            threshold_lr_lower : 1e-3,
            threshold_pruning : 1e-2,
        };

        SimConfig {
            steps : 100,
            num_targets : 5,
            fov : ((0.0,1e3),(0.0,1e3)),
            config : config,
        }
    }
}
