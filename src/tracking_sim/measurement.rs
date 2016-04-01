use rm::linalg::matrix::Matrix;

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

    pub fn copy(&self) -> Measurement {
        Measurement {
            data : self.data.select_cols(&[0]),
        }
    }
}

