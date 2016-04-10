use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Config;
use filter::tracking_sim::Manager;
use filter::tracking_sim::Track;
use filter::tracking_sim::Measurement;

use rm::linalg::matrix::Matrix;



#[test]
fn process_single_measurement() {
    let mut config = Config::new();
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let target = Target::new(init_state,2.5,0.0);
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut sensor = Sensor::new(config.clone());
    let z = sensor.measure(&target);
    assert!(z.len() == 1);

    let mut manager = Manager::new(config.clone());
    manager.process(z);
    assert!(manager.tracks.len() == 1);
}

#[test]
fn measurement_association() {
    let mut config = Config::new();
    let init_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_state2 = Matrix::<f64>::new(4,1,vec![-10.0, -10.0, 2.0, 3.0]);
    let init_state3 = Matrix::<f64>::new(4,1,vec![-50.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::identity(4) * 5.0;
    let track1 = Track::new(&init_state1,&init_covar,1.0);
    let track2 = Track::new(&init_state2,&init_covar,1.0);
    let track3 = Track::new(&init_state3,&init_covar,1.0);
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut manager = Manager::new(config.clone());
    manager.tracks = vec![track1,track2,track3];

    //test msr
    let msr1 = Measurement::new(vec![11.0,11.0]);
    let msr2 = Measurement::new(vec![-11.0,-11.0]);
    let msr3 = Measurement::new(vec![-48.0,12.0]);
    let msrs = vec![msr1,msr2,msr3];

    let a = manager.associate(&msrs);
    assert!(a.len() == 3);
    for i in a.iter() {
        assert!(i.len() == 1);
    }
}


#[test]
fn measurement_non_association() {
    let mut config = Config::new();
    let init_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_state2 = Matrix::<f64>::new(4,1,vec![-10.0, -10.0, 2.0, 3.0]);
    let init_state3 = Matrix::<f64>::new(4,1,vec![-50.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::identity(4) * 5.0;
    let track1 = Track::new(&init_state1,&init_covar,1.0);
    let track2 = Track::new(&init_state2,&init_covar,1.0);
    let track3 = Track::new(&init_state3,&init_covar,1.0);
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut manager = Manager::new(config.clone());
    manager.tracks = vec![track1,track2,track3];

    //test msr
    let msr1 = Measurement::new(vec![11.0,11.0]);
    let msr2 = Measurement::new(vec![-11.0,-11.0]);
    let msr3 = Measurement::new(vec![-48.0,12.0]);
    let msr4 = Measurement::new(vec![423218.0,312312312.0]);
    let msrs = vec![msr1,msr2,msr3,msr4];

    let a = manager.get_non_assoc(&msrs);
    assert!(a.len() == 1);

}

#[test]
fn likelihood_ratio() {
    let mut config = Config::new();
    config.p_D = 0.95;
    config.rho_F = 1e-6;
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::ones(4,4) + Matrix::<f64>::identity(4) * 50.0;
    let track1 = Track::new(&init_state,&init_covar,1.0);
    let mut manager = Manager::new(config.clone());
    manager.tracks = vec![track1];
    let msr1 = Measurement::new(vec![8.0, 13.0]);
    let msr2 = Measurement::new(vec![18.0, 10.0]);
    let msr3 = Measurement::new(vec![5.0, 15.0]);
    let msrs = vec![msr1,msr2,msr3];
    manager.process(msrs);

    let lr = manager.tracks.first().unwrap().lr;
    assert!((lr - 5.9133e+03).abs() < 1_f64);
    // matlab code for testing
    /*
    p_D = 0.95
    rho_F = 1e-6
    x0 = [10 10 2 3]'
    P0 = ones(4,4) + eye(4) * 50
    z = [8 13; 18 10; 5 15]'
    R = eye(2)
    H = kron([1 0], eye(2))
    S = H * P0 * H' + R
    W = P0 * H' * S^-1
    P = P0 - W*S*W'
    x1 = x0 + W*(z(:,1) - H*x0)
    x2 = x0 + W*(z(:,2) - H*x0)
    x3 = x0 + W*(z(:,3) - H*x0)
    w0 = 1-p_D
    w1 = p_D / rho_F * mvnpdf(z(:,1),H * x0, S)
    w2 = p_D / rho_F * mvnpdf(z(:,2),H * x0, S)
    w3 = p_D / rho_F * mvnpdf(z(:,3),H * x0, S)
    lr = w0+w1+w2+w3
    */
}