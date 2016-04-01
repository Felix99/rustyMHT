use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Filter;
use filter::tracking_sim::Config;
use filter::tracking_sim::Track;
use filter::tracking_sim::Hypothesis;
use filter::tracking_sim::Measurement;
use rm::linalg::matrix::Matrix;
use std::f64;


#[test]
fn Kalman_prediction() {
    let config = Config::new();
    let mut filter = Filter::new();
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let mut track = Track::new(&init_state,&config.init_covar,1.0);
    filter.predict(&mut track);
    let final_state = Matrix::<f64>::new(4,1,vec![12.0, 13.0, 2.0, 3.0]);
    let final_covar = Matrix::<f64>::new(4,4,vec![200.3, 0.0, 100.5, 0.0,
        0.0, 200.3, 0.0, 100.5,
        100.5, 0.0, 101.0, 0.0,
        0.0, 100.5, 0.0, 101.0]);
    let res = &track.state - &final_state;
    for i in res.data() {
        assert!(i.abs() < 1e-3);
    }
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.data() {
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
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::ones(4,4) + Matrix::<f64>::identity(4) * 50.0;
    let mut track = Track::new(&init_state,&init_covar,1.0);

//    let msr = Measurement {data : Matrix::<f64>::new(vec![8.0, 13.0]).reshape(&[2,1])};
    let msr = Measurement::new(vec![8.0, 13.0]);
    filter.update(&mut track, &msr);

    let final_state = Matrix::<f64>::new(4,1,vec![8.0396, 12.9415, 2.0189, 3.0189]);
    let final_covar = Matrix::<f64>::new(4,4,vec![
    0.9808,    0.0004,    0.0189,    0.0189,
    0.0004,    0.9808,    0.0189,    0.0189,
    0.0189,    0.0189,   50.9623,    0.9623,
    0.0189,    0.0189,    0.9623,   50.9623]);
    let diff_state = &final_state - &track.state;
    for i in diff_state.data() {
        assert!(i.abs() < 1e-3);
    }
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.data() {
        assert!(i.abs() < 1e-3);
    }
}

#[test]
fn Hypotheses_merged() {
    let init_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_state2 = Matrix::<f64>::new(4,1,vec![20.0, 25.0, -1.0, 2.0]);
    let init_covar1 = Matrix::<f64>::identity(4) * 100.0;
    let init_covar2 = Matrix::<f64>::identity(4) * 75.0;
    let weight1 = 0.4;
    let weight2 = 0.6;
    let mut track = Track::new(&init_state1,&init_covar1,1.0);
    track.hypotheses = vec![Hypothesis::new(&init_state1,&init_covar1,weight1),
        Hypothesis::new(&init_state2,&init_covar2,weight2)];

    track.update_state();

    let final_state = Matrix::<f64>::new(4,1,vec![16.0, 19.0, 0.2, 2.4]);
    let final_covar = Matrix::<f64>::new(4,4,vec![
    109.0000,   36.0000,   -7.2000,   -2.4000,
    36.0000,  139.0000,  -10.8000,   -3.6000,
    -7.2000,  -10.8000,   87.1600,    0.7200,
    -2.4000,   -3.6000,    0.7200,   85.2400]);

    // compare state
    let diff_state = &final_state - &track.state;
    for i in diff_state.data() {
        assert!(i < &0.01);
    }
    // compare covar
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.data() {
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
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::ones(4,4) + Matrix::<f64>::identity(4) * 50.0;
    let mut track = Track::new(&init_state,&init_covar,1.0);

    let msr1 = Measurement::new(vec![8.0, 13.0]);
    let msr2 = Measurement::new(vec![18.0, 10.0]);
    let msr3 = Measurement::new(vec![5.0, 15.0]);
    let msrs = vec![msr1,msr2,msr3];
    filter.update_mht(&mut track, &msrs);

    let final_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let final_state2 = Matrix::<f64>::new(4,1,vec![8.0396, 12.9415, 2.0189, 3.0189]);
    let final_state3 = Matrix::<f64>::new(4,1,vec![17.8461,10.0030, 2.1509, 3.1509]);
    let final_state4 = Matrix::<f64>::new(4,1,vec![5.0980, 14.9020, 2.0000, 3.0000]);
    let final_states = vec![final_state1,final_state2,final_state3,final_state4];
    let final_weights = vec![0.000008, 0.433033, 0.265726, 0.301233];
    let final_covar = Matrix::<f64>::new(4,4,vec![
    0.9808,    0.0004,    0.0189,    0.0189,
    0.0004,    0.9808,    0.0189,    0.0189,
    0.0189,    0.0189,   50.9623,    0.9623,
    0.0189,    0.0189,    0.9623,   50.9623]);
    let final_covars = vec![init_covar,final_covar];

    assert!(&track.hypotheses.len() == &4);
    // state
    for h in track.hypotheses.iter() {
        let mut dist = f64::INFINITY;
        for state in final_states.iter() {
            let dist_i = state - &h.state;

            let d_i = (dist_i.transpose() * &dist_i)[[0,0]];
            if d_i < dist {
                dist = d_i;
            }
        }
        assert!(dist < 1e-2);
    }

    // check weights
    for h in track.hypotheses.iter() {
        let mut dist = f64::INFINITY;
        for weight in final_weights.iter() {
            let d_i = weight - h.weight;
            if d_i < dist {
                dist = d_i;
            }
        }
        assert!(dist < 1e-2);
    }

    // check covariances
    for h in track.hypotheses.iter() {
        let mut dist = f64::INFINITY;
        for covar in final_covars.iter() {
            let diff = covar - &h.covar;
            let d_i = diff.data().iter().fold(0.0, |a,e| a + e.powf(2.0));
            if d_i < dist {
                dist = d_i;
            }
        }
        assert!(dist < 1e-2);
    }



    /* Matlab code for checking.
    p_D = 0.95
    rho_F = 1e-6
    x0 = [10 10 2 3]';
    P0 = ones(4,4) + eye(4) * 50;
    z = [8 13; 18 10; 5 15]'
    R = eye(2)
    H = kron([1 0], eye(2))
    S = H * P0 * H' + R;
    W = P0 * H' * S^-1;
    P = P0 - W*S*W'
    x1 = x0 + W*(z(:,1) - H*x0)
    x2 = x0 + W*(z(:,2) - H*x0)
    x3 = x0 + W*(z(:,3) - H*x0)
    w0 = 1-p_D;
    w1 = p_D / rho_F * mvnpdf(z(:,1),H * x0, S)
    w2 = p_D / rho_F * mvnpdf(z(:,2),H * x0, S)
    w3 = p_D / rho_F * mvnpdf(z(:,3),H * x0, S)
    sum = w0+w1+w2+w3
    w0 = w0 / sum
    w1 = w1 / sum
    w2 = w2 / sum
    w3 = w3 / sum
    fprintf('%f, %f, %f, %f\n',w0,w1,w2,w3)
        */
}

#[test]
fn normalize_weights() {
    let init_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_state2 = Matrix::<f64>::new(4,1,vec![20.0, 25.0, -1.0, 2.0]);
    let init_covar1 = Matrix::<f64>::identity(4) * 100.0;
    let init_covar2 = Matrix::<f64>::identity(4) * 75.0;
    let weight1 = 3.4;
    let weight2 = 2.6;
    let mut track = Track::new(&init_state1,&init_covar1,1.0);
    track.hypotheses = vec![Hypothesis::new(&init_state1,&init_covar1,weight1),
        Hypothesis::new(&init_state2,&init_covar2,weight2)];

    track.normalize_weights();

    let sum = track.hypotheses[0].weight + track.hypotheses[1].weight;
    assert!(sum - 1.0 <1e-5);

}


#[test]
fn gating() {
    let mut config = Config::new();
    config.p_D = 0.9;
    config.rho_F = 1e-5;
    let mut filter = Filter::new();
    filter.set_config(config);
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_covar = Matrix::<f64>::ones(4,4) + Matrix::<f64>::identity(4) * 50.0;
    let mut track = Track::new(&init_state,&init_covar,1.0);

    let msr1 = Measurement::new(vec![8.0, 13.0]);
    let msr2 = Measurement::new(vec![18.0, 10.0]);
    let msr3 = Measurement::new(vec![5.0, 15.0]);
    let msr4 = Measurement::new(vec![25.0, -15.0]);
    let msr5 = Measurement::new(vec![-5.0, 35.0]);
    let msrs = vec![msr1,msr2,msr3];
    let gated = filter.gate(&track, &msrs);

    assert!(gated.len() == 3);
}

#[test]
fn hypothesis_merging() {
    let mut config = Config::new();
    config.p_D = 0.9;
    config.rho_F = 1e-5;
    let mut filter = Filter::new();
    filter.set_config(config);
    let init_covar1 = Matrix::<f64>::ones(4,4) + Matrix::<f64>::identity(4) * 50.0;
    let init_covar2 = Matrix::<f64>::new(4,4,vec![
    0.9808,    0.0004,    0.0189,    0.0189,
    0.0004,    0.9808,    0.0189,    0.0189,
    0.0189,    0.0189,   50.9623,    0.9623,
    0.0189,    0.0189,    0.9623,   50.9623]);

    let init_state1 = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let init_state2 = Matrix::<f64>::new(4,1,vec![8.0396, 12.9415, 2.0189, 3.0189]);
    let init_state3 = Matrix::<f64>::new(4,1,vec![17.8461,10.0030, 2.1509, 3.1509]);
    let init_state4 = Matrix::<f64>::new(4,1,vec![5.0980, 14.9020, 2.0000, 3.0000]);
    let weights = vec![0.000008, 0.433033, 0.265726, 0.301233];

    let h1 = Hypothesis::new(&init_state1,&init_covar1,weights[0]);
    let h2 = Hypothesis::new(&init_state2,&init_covar2,weights[1]);
    let h3 = Hypothesis::new(&init_state3,&init_covar1,weights[2]);
    let h4 = Hypothesis::new(&init_state4,&init_covar2,weights[3]);

    let mut track = Track::new(&init_state1,&init_covar1,1.0);
    track.hypotheses = vec![h1,h2,h3,h4];

    track.merge_hypotheses();
    assert!(track.hypotheses.len() < 4);
    let weights_new : f64 = track.hypotheses.iter().fold(0_f64, |a,e| a + e.weight);
    assert!((&weights_new-1.0_f64).abs() < 1e-4);

}

