// Cymbalum, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license

use rand::distributions::{Sample, Range};
use rand::Rng;

use std::f64;
use std::mem;

use super::MCMove;

use types::{Matrix3, One};
use system::{System, EnergyCache};

/// Monte-Carlo move for rescaling the simulation cell
pub struct Resize {
    /// Delta for translation of the box length
    delta: f64,
    /// Range of translation delta
    delta_range: Range<f64>,
    /// system state before resizing
    old_system: System,
    /// target pressure
    pressure: f64,
}

impl Resize {
    /// Create a new `Resize` move, with target pressure `pressure` and maximum
    /// displacement of `max_delta`.
    pub fn new(pressure: f64, max_delta: f64) -> Resize {
        assert!(max_delta > 0.0, "max_delta must be positive in Resize move");
        Resize {
            delta: 0.,
            delta_range: Range::new(-max_delta, max_delta),
            old_system: System::new(),
            pressure: pressure,
        }
    }
}

impl MCMove for Resize {
    fn describe(&self) -> &str {
        "resizing of the cell"
    }

    fn prepare(&mut self, system: &mut System, rng: &mut Box<Rng>) -> bool {
        self.delta = self.delta_range.sample(rng);
        self.old_system = system.clone();
        let volume = system.volume();
        let scaling_factor = f64::cbrt((volume + self.delta) / volume);
        system.cell_mut().scale_mut(Matrix3::one() * scaling_factor);
        let new_cell = system.cell().clone();
        for particle in system.iter_mut() {
            let fractional = self.old_system.cell().fractional(&particle.position);
            particle.position = new_cell.cartesian(&fractional);
        }
        return true;
    }

    fn cost(&self, system: &System, beta: f64, cache: &mut EnergyCache) -> f64 {
        cache.unused();
        let old_energy = self.old_system.potential_energy();
        let new_energy = system.potential_energy();
        let delta_energy = new_energy - old_energy;

        let new_volume = system.volume();
        let old_volume = self.old_system.volume();
        let delta_volume = new_volume - old_volume;

        let cost = beta * (delta_energy + self.pressure * delta_volume)
                   - (system.size() as f64) * f64::ln(new_volume / old_volume);
        cost
    }

    fn apply(&mut self, _: &mut System) {
        // Nothing to do
    }

    fn restore(&mut self, system: &mut System) {
        mem::swap(system, &mut self.old_system);
    }
}
