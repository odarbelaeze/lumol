// Lumol, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license
use toml::Value;

use lumol::sys::System;
use lumol::energy::{Wolf, Ewald, CoulombicPotential};

use error::{Error, Result};
use FromToml;
use super::read_restriction;
use super::InteractionsInput;

impl InteractionsInput {
    /// Read the "coulomb" section from the potential configuration. This is an
    /// internal function, public because of the code organization.
    // TODO: use restricted privacy here
    pub fn read_coulomb(&self, system: &mut System) -> Result<()> {
        let coulomb = match self.config.get("coulomb") {
            Some(coulomb) => coulomb,
            None => return Ok(())
        };

        let coulomb = try!(coulomb.as_table().ok_or(
            Error::from("The 'coulomb' section must be a table")
        ));

        let solvers = coulomb.keys().cloned()
                             .filter(|key| key != "restriction")
                             .collect::<Vec<_>>();

        if solvers.len() != 1 {
            return Err(Error::from(
                format!("Got more than one coulombic solver: {}", solvers.join(" - "))
            ));
        }

        let key = &*solvers[0];
        if let Value::Table(ref table) = coulomb[key] {
            let mut potential: Box<CoulombicPotential> = match key {
                "wolf" => Box::new(try!(Wolf::from_toml(table))),
                "ewald" => Box::new(try!(Ewald::from_toml(table))),
                other => {
                    return Err(Error::from(format!("Unknown coulomb solver '{}'", other)))
                },
            };

            if let Some(restriction) = try!(read_restriction(coulomb)) {
                potential.set_restriction(restriction);
            }

            system.interactions_mut().set_coulomb(potential);
            Ok(())
        } else {
            Err(
                Error::from(format!("Coulombic solver '{}' must be a table", key))
            )
        }
    }

    /// Read the "charges" from the potential configuration. This is an internal
    /// function, public because of the code organization.
    // TODO: use restricted privacy here
    pub fn read_charges(&self, system: &mut System) -> Result<()> {
        let charges = match self.config.get("charges") {
            Some(charges) => charges,
            None => return Ok(())
        };

        let charges = try!(charges.as_table().ok_or(
            Error::from("The 'charges' section must be a table")
        ));

        let mut total_charge = 0.0;
        for (name, charge) in charges.iter() {
            let charge = match *charge {
                Value::Integer(val) => val as f64,
                Value::Float(val) => val,
                _ => {
                    return Err(Error::from("Charges must be numbers"));
                }
            };

            let mut nchanged = 0;
            for particle in system.iter_mut() {
                if particle.name() == name {
                    particle.charge = charge;
                    nchanged += 1;
                    total_charge += charge;
                }
            }

            if nchanged == 0 {
                warn!("No particle with name '{}' was found while setting the charges", name);
            } else {
                info!("Charge set to {:+} for {} {} particles", charge, nchanged, name);
            }
        }

        if total_charge.abs() > 1e-6 {
            warn!("System is not neutral and have a net charge of {:+}", total_charge);
        }
        Ok(())
    }
}


    // #[test]
    // fn ewald() {
    //     let path = Path::new(file!()).parent().unwrap().join("data").join("ewald.toml");
    //     let mut system = System::new();
    //     system.add_particle(Particle::new("A"));
    //     system.add_particle(Particle::new("B"));
    //
    //     let input = InteractionsInput::new(path).unwrap();
    //     input.read(&mut system).unwrap();
    //
    //     assert!(system.interactions().coulomb().is_some());
    //     assert_eq!(system[0].charge, -8.0);
    //     assert_eq!(system[1].charge, 3.0);
    // }
    //
    // #[test]
    // fn wolf() {
    //     let mut system = System::new();
    //     system.add_particle(Particle::new("A"));
    //     system.add_particle(Particle::new("B"));
    //
    //     let path = Path::new(file!()).parent().unwrap().join("data").join("wolf.toml");
    //     let input = InteractionsInput::new(path).unwrap();
    //     input.read(&mut system).unwrap();
    //
    //     assert!(system.interactions().coulomb().is_some());
    //     assert_eq!(system[0].charge, -2.0);
    //     assert_eq!(system[1].charge, 2.0);
    // }
