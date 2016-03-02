// Cymbalum, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license

//! **cymbalum** is a molecular simulation library, which provides basic
//! building blocks to create your very own simulations. It is based on two main
//! ideas:
//!
//!  - any algorithm used in the simulation can be replaced by another one. This
//!    allow for modularity and easy developement of novel algorithms.
//!  - data and algorithms should be separated. This is accomplished through the
//!    use of the `System` (for data) and `Simulation` (for algorithms) types,
//!    interacting together to run the simulation on some data.
//!
//!  Rust provides a nice way to implement these two ideas with the concept of
//!  traits.

#![allow(non_snake_case)]
#![warn(
    missing_docs,
    trivial_casts,
    unused_import_braces,
    variant_size_differences,
    unused_qualifications
)]

#![cfg_attr(feature="lint", feature(plugin))]
#![cfg_attr(feature="lint", plugin(clippy))]
#![cfg_attr(feature="lint", warn(clippy))]
#![cfg_attr(feature="lint", allow(needless_return, needless_range_loop))]

#[macro_use]
extern crate log;
#[macro_use]
extern crate bitflags;

extern crate chemfiles;
extern crate ndarray;
extern crate num;
extern crate rand;
extern crate special;
extern crate yaml_rust as yaml;

#[macro_use]
mod tests;
#[macro_use]
mod utils;

mod logging;
pub use logging::{Logger, LogLevel};

pub mod units;
pub mod constants;

pub mod types;
pub mod potentials;
pub mod system;
pub mod simulation;

pub use types::*;
pub use potentials::*;
pub use system::*;
pub use simulation::*;

pub mod input;
