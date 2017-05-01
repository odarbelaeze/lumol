use std::ops::Range;

use rayon::prelude::*;
use rayon::iter::{Map, MapFn};


/// Utility trait that adds shortcuts for `IntoParallelIterator` structs.
pub trait ParallelShortcuts : IntoParallelIterator + Sized {

    /// Shortcut for `into_par_iter().map()`
    fn par_map<F, R>(self, map_op: F) -> Map<Self::Iter, MapFn<F>>
        where F: Fn(Self::Item) -> R + Sync, R: Send{
        self.into_par_iter().map(map_op)
    }
}

impl<T> ParallelShortcuts for T where T : IntoParallelIterator {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map() {
        let s = (0..100_i32).par_map(|i| -i).sum();
        assert_eq!(-4950, s);
    }
}