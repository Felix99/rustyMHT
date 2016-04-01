use rm::linalg::matrix::Matrix;

use filter::tracking_sim::Target;
use filter::tracking_sim::Dynamics;



#[test]
fn main() {
    let initState = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let finalState = Matrix::<f64>::new(4,1,vec![15.0, 17.5, 2.0, 3.0]);
    let mut target = Target::new(initState,2.5,0.0);
    target.move_forward();
    let res = &target.state - &finalState;
    for i in res.data() {
        assert!(i.abs() < 1e-3);
    }
}


