
pub trait Merger<T: IsSimilar + Clone> {
    // to be implemented
    fn merge_all(&self, to_merge: Vec<T>) -> T;

    fn merge(&self, stack: &Vec<T>, threshold : f64) -> Vec<T> {
        let merged : Vec<T> = Vec::new();
        let result = self.merge_local(&merged, &stack, threshold);
        result
    }

    fn merge_local(&self, merged: &Vec<T>, stack: &Vec<T>, threshold : f64) -> Vec<T> {
        if stack.is_empty() {
            return merged.clone();
        } else {
            let (to_merge, not_to_merge) = self.find_elements_similar_to_first(stack, threshold);
            let mut new_merged = merged.clone();
            new_merged.push(self.merge_all(to_merge));
            return self.merge_local(&new_merged, &not_to_merge, threshold);
        }

    }

    fn find_elements_similar_to_first(&self, stack: &Vec<T>, threshold : f64) -> (Vec<T>,Vec<T>) where T: IsSimilar + Clone {
        let first = &stack[0];
        let mut to_merge : Vec<T> = Vec::new();
        let mut not_to_merge : Vec<T> = Vec::new();
        for e in stack.into_iter() {
            if e.is_similar(first, threshold) {
                to_merge.push(e.clone())
            } else {
                not_to_merge.push(e.clone())
            }
        }
        (to_merge, not_to_merge)
    }
}

pub trait IsSimilar {
    fn is_similar(&self, other: &Self, threshold: f64) -> bool;
}
