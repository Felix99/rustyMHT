use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Filter;
use filter::tracking_sim::Config;
use filter::tracking_sim::Track;
use filter::tracking_sim::Hypothesis;
use filter::tracking_sim::Measurement;
use numeric::Tensor;
use std::f64;


#[test]
fn Kalman_prediction() {
    let config = Config::new();
    let mut filter = Filter::new();
    let init_state = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let mut track = Track::new(&init_state,&config.init_covar,1.0);
    filter.predict(&mut track);
    let final_state = Tensor::<f64>::new(vec![12.0, 13.0, 2.0, 3.0]).reshape(&[4,1]);
    let final_covar = Tensor::<f64>::new(vec![200.3, 0.0, 100.5, 0.0,
        0.0, 200.3, 0.0, 100.5,
        100.5, 0.0, 101.0, 0.0,
        0.0, 100.5, 0.0, 101.0]).reshape(&[4,4]);
    assert!(&track.state == &final_state);
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.slice() {
        assert!(i < &0.01);
    }
}

#[test]
fn Kalman_filtering() {
    let mut config = Config::new();
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut filter = Filter::new();
    filter.set_config(config);
    let init_state = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let init_covar = Tensor::<f64>::ones(&[4,4]) + Tensor::<f64>::eye(4) * 50.0;
    let mut track = Track::new(&init_state,&init_covar,1.0);

//    let msr = Measurement {data : Tensor::<f64>::new(vec![8.0, 13.0]).reshape(&[2,1])};
    let msr = Measurement::new(vec![8.0, 13.0]);
    filter.update(&mut track, &msr);

    let final_state = Tensor::<f64>::new(vec![8.0396, 12.9415, 2.0189, 3.0189]).reshape(&[4,1]);
    let final_covar = Tensor::<f64>::new(vec![
    0.9808,    0.0004,    0.0189,    0.0189,
    0.0004,    0.9808,    0.0189,    0.0189,
    0.0189,    0.0189,   50.9623,    0.9623,
    0.0189,    0.0189,    0.9623,   50.9623]).reshape(&[4,4]);
    let diff_state = &final_state - &track.state;
    for i in diff_state.slice() {
        assert!(i.abs() < 1e-3);
    }
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.slice() {
        assert!(i.abs() < 1e-3);
    }
}

#[test]
fn Hypotheses_merged() {
    let init_state1 = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let init_state2 = Tensor::<f64>::new(vec![20.0, 25.0, -1.0, 2.0]).reshape(&[4,1]);
    let init_covar1 = Tensor::<f64>::eye(4) * 100.0;
    let init_covar2 = Tensor::<f64>::eye(4) * 75.0;
    let weight1 = 0.4;
    let weight2 = 0.6;
    let mut track = Track::new(&init_state1,&init_covar1,1.0);
    track.hypotheses = vec![Hypothesis::new(&init_state1,&init_covar1,weight1),
        Hypothesis::new(&init_state2,&init_covar2,weight2)];

    track.update_state();

    let final_state = Tensor::<f64>::new(vec![16.0, 19.0, 0.2, 2.4]).reshape(&[4,1]);
    let final_covar = Tensor::<f64>::new(vec![
    109.0000,   36.0000,   -7.2000,   -2.4000,
    36.0000,  139.0000,  -10.8000,   -3.6000,
    -7.2000,  -10.8000,   87.1600,    0.7200,
    -2.4000,   -3.6000,    0.7200,   85.2400]).reshape(&[4,4]);

    // compare state
    let diff_state = &final_state - &track.state;
    for i in diff_state.slice() {
        assert!(i < &0.01);
    }
    // compare covar
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.slice() {
        assert!(i < &0.01);
    }
}

#[test]
fn multi_hypotheses_update() {
    let mut config = Config::new();
    config.p_D = 0.9;
    config.rho_F = 1e-5;
    let mut filter = Filter::new();
    filter.set_config(config);
    let init_state = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let init_covar = Tensor::<f64>::ones(&[4,4]) + Tensor::<f64>::eye(4) * 50.0;
    let mut track = Track::new(&init_state,&init_covar,1.0);

    let msr1 = Measurement::new(vec![8.0, 13.0]);
    let msr2 = Measurement::new(vec![18.0, 10.0]);
    let msr3 = Measurement::new(vec![5.0, 15.0]);
    let msrs = vec![msr1,msr2,msr3];
    filter.update_mht(&mut track, &msrs);

    let final_state1 = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let final_state2 = Tensor::<f64>::new(vec![8.0396, 12.9415, 2.0189, 3.0189]).reshape(&[4,1]);
    let final_state3 = Tensor::<f64>::new(vec![17.8461,10.0030, 2.1509, 3.1509]).reshape(&[4,1]);
    let final_state4 = Tensor::<f64>::new(vec![5.0980, 14.9020, 2.0000, 3.0000]).reshape(&[4,1]);
    let final_states = vec![final_state1,final_state2,final_state3,final_state4];
    let final_covar = Tensor::<f64>::new(vec![
    0.9808,    0.0004,    0.0189,    0.0189,
    0.0004,    0.9808,    0.0189,    0.0189,
    0.0189,    0.0189,   50.9623,    0.9623,
    0.0189,    0.0189,    0.9623,   50.9623]).reshape(&[4,4]);

    assert!(&track.hypotheses.len() > &0);
    for h in track.hypotheses.iter() {
        let mut dist = f64::INFINITY;
        for state in final_states.iter() {
            let dist_i = state - &h.state;
            let d_i = dist_i.transpose().dot(&dist_i).slice()[0].sqrt();
            if d_i < dist {
                dist = d_i;
            }
        }
        assert!(dist < 1e-2);
    }



    /* Matlab code for checking.
    x0 = [10 10 2 3]';
    P0 = ones(4,4) + eye(4) * 50;
    z = [8 13; 18 10; 5 15]'
    R = eye(2)
    H = kron(eye(2),[1 0])
    H = kron([1 0], eye(2))
    S = H * P0 * H' + R;
    W = P * H' * S^-1;
    W = P0 * H' * S^-1;
    P = P0 - W*S*W'
    x0 + W(z(:,1) - H*x0)
    x0 + W*(z(:,1) - H*x0)
    x0 + W*(z(:,2) - H*x0)
    x0 + W*(z(:,3) - H*x0)
        */
}



