#[macro_use(tensor)]
use numeric::random::RandomState;
use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Target;
use tracking_sim::Config;
use tracking_sim::Measurement;

pub struct Sensor {
    config : Config,
    la : Linalg,
}

impl Sensor {
    pub fn new(config: Config) -> Sensor {
        Sensor {
            config : config,
            la : Linalg::new(),
        }
    }

    pub fn measure(&mut self, target: &Target) -> Vec<Measurement> {
        if self.config.rho_F > 0.0 {
            let mut fa = self.false_measurements();
            for z in self.target_measurement(target).iter() {
                let z_copy = Measurement {
                    data: self.la.copy(&z.data),
                };
                fa.push(z_copy);
            }
            fa
        } else {
            self.target_measurement(target)
        }
    }

    fn target_measurement(&mut self, target: &Target) -> Vec<Measurement> {
        let mut res = Vec::new();
        if self.target_detection() {
            let z = self.la.mvnrnd(&(self.config.msr_matrix.dot(&target.state)), &self.config.msr_covar);
            res.push(Measurement{data:z.reshape(&[2,1])})
        }
        res
    }

    fn target_detection(&mut self) -> bool {
        let t = self.la.uniform(0.0, 1.0);
        if t < self.config.p_D {
            true
        } else {
            false
        }
    }

    fn false_measurements(&mut self) -> Vec<Measurement> {
        let x_size = (self.config.fov.unwrap().0).1 - (self.config.fov.unwrap().0).0;
        let y_size = (self.config.fov.unwrap().1).1 - (self.config.fov.unwrap().1).0;
        let fov_size = x_size * y_size;
        let lambda = &fov_size * self.config.rho_F;
        let num_fa = self.la.poisson(lambda);
        let mut fa = Vec::new();
        for _ in 0..num_fa {
            let x = self.la.uniform((self.config.fov.unwrap().0).0,(self.config.fov.unwrap().0).1);
            let y = self.la.uniform((self.config.fov.unwrap().1).0,(self.config.fov.unwrap().1).1);
            fa.push(Measurement::new(vec![x,y]));
        }
        fa
    }

}
