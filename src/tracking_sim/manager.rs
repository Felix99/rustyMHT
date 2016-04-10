use tracking_sim::Config;
use tracking_sim::Track;
use tracking_sim::Measurement;
use tracking_sim::Filter;



pub struct Manager {
    pub tracks : Vec<Track>,
    pub filter : Filter,
    current_track_id : i64,
    pub config : Config,
    dim : usize,
    time : f64,
}

impl Manager {
    pub fn new(config: Config) -> Manager {
        Manager {
            tracks : Vec::new(),
            filter : Filter::new(config.clone()),
            current_track_id : 1,
            config : config.clone(),
            dim : config.msr_matrix.cols(),
            time : 1_f64,
        }
    }


    pub fn process(&mut self, msrs: Vec<Measurement>) {
        // associate
        let assoc = self.associate(&msrs);
        let non_assoc = self.get_non_assoc(&msrs);

        // filter
        for i in 0..self.tracks.len() {
            let mut t = &mut self.tracks[i];
            self.filter.update_mht(t,&assoc[i]);
        }

        // generate new tracks
        {
            for z in non_assoc.iter() {
                let t = self.generate_new_track(z);
                self.tracks.push(t);
            }
        }
    }

    pub fn generate_new_track(&self, z: &Measurement) -> Track {
        let state = &self.config.msr_matrix.transpose() * &z.data;
        Track::new(&state, &self.config.init_covar, self.time)
    }

    pub fn associate(&self, msrs: &Vec<Measurement>) -> Vec<Vec<Measurement>> {
        self.tracks.iter().map(|t| self.filter.gate(t,msrs)).collect()
    }

    pub fn get_non_assoc(&self, msrs: &Vec<Measurement>) -> Vec<Measurement> {
        let mut non_assoc = Vec::<Measurement>::new();
        for z in msrs {
            let mut not_associated = true;
            for t in &self.tracks {
                let gate = self.filter.gate(&t,&vec![z.clone()]);
                if !gate.is_empty() {
                    not_associated = false;
                    break;
                }
            }
            if not_associated {
                non_assoc.push(z.clone());
            }
        }
        non_assoc
    }
}
