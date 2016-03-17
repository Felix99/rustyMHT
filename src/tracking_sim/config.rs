use numeric::Tensor;

pub struct Config {
    pub msr_covar: Tensor<f64>,
    pub msr_matrix: Tensor<f64>,
    pub q : f64,
    pub delta_t : f64,
    pub init_covar : Tensor<f64>,
    pub p_D : f64,
    pub rho_F : f64,
    pub mu_gating : f64,
    pub fov : Option<((f64,f64),(f64,f64))>,
}

impl Config {
    pub fn new() -> Config {
        Config {
            msr_covar : Tensor::<f64>::eye(2),
            msr_matrix : Tensor::<f64>::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(&[2,4]),
            q : 1.0,
            delta_t : 1.0,
            init_covar :  Tensor::<f64>::eye(4) * 100.0,
            p_D : 0.95,
            rho_F : 1e-6,
            mu_gating : 9.21,
            fov : None,
        }
    }
}