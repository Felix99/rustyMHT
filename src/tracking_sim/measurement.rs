use rm::linalg::Matrix;

pub struct Measurement {
    pub data : Matrix<f64>,
}

impl Measurement {
    pub fn new(msrs: Vec<f64>) -> Measurement {
        let dim = msrs.len();
        Measurement {
            data : Matrix::<f64>::new(dim,1,msrs),
        }
    }

}

impl Clone for Measurement {
    fn clone(&self) -> Measurement {
        Measurement { data : self.data.clone() }
    }
}

