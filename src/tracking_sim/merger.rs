
pub trait Merger {
    // to be implemented
    fn merge_all<T>(&self, to_merge: Vec<T>) -> T;

    fn merge<T>(&self, stack: &Vec<T>) -> Vec<T> where T: IsSimilar + Clone {
        let merged : Vec<T> = Vec::new();
        let result = self.merge_local(&merged, &stack);
        result
    }

    fn merge_local<T>(&self, merged: &Vec<T>, stack: &Vec<T>) -> Vec<T> where T: IsSimilar + Clone {
        if stack.is_empty() {
            return merged.clone();
        } else {
            let (to_merge, not_to_merge) = self.find_elements_similar_to_first(stack);
            let mut new_merged = merged.clone();
            new_merged.push(self.merge_all(to_merge));
            return self.merge_local(&new_merged, &not_to_merge);
        }

    }

    fn find_elements_similar_to_first<T>(&self, stack: &Vec<T>) -> (Vec<T>,Vec<T>) where T: IsSimilar + Clone {
        let first = &stack[0];
        let mut to_merge : Vec<T> = Vec::new();
        let mut not_to_merge : Vec<T> = Vec::new();
        for e in stack.into_iter() {
            if e.is_similar(first) {
                to_merge.push(e.clone())
            } else {
                not_to_merge.push(e.clone())
            }
        }
        (to_merge, not_to_merge)
    }
}

pub trait IsSimilar {
    fn is_similar(&self, other: &Self) -> bool;
}
