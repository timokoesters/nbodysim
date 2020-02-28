//! This module is responsible for defining the external config format and parsing it.

use {
    crate::{galaxygen, Particle},
    serde::Deserialize,
};

#[derive(Deserialize, Clone, Debug)]
/// The configuration that specifies the initial values of the simulation.
pub struct Config {
    pub camera_pos: [f32; 3],
    pub safety: f64,
    pub constructions: Vec<Construction>,
}

#[derive(Deserialize, Clone, Debug)]
/// Description of a (group of) particles.
pub enum Construction {
    Particle {
        pos: [f32; 3],
        vel: [f32; 3],
        mass: f64,
    },
    Galaxy {
        center_pos: [f32; 3],
        center_vel: [f32; 3],
        center_mass: f64,
        amount: u32,
        normal: [f32; 3],
    },
}

impl Config {
    /// Build the actual particles from the constructions.
    pub fn construct_particles(&self) -> Vec<Particle> {
        let mut particles = Vec::new();

        // Those with mass first
        for c in &self.constructions {
            particles.push(match c {
                Construction::Particle { pos, vel, mass } => {
                    Particle::new((*pos).into(), (*vel).into(), *mass, 1.0)
                }
                Construction::Galaxy {
                    center_pos,
                    center_vel,
                    center_mass,
                    ..
                } => Particle::new(
                    (*center_pos).into(),
                    (*center_vel).into(),
                    *center_mass,
                    1.0,
                ),
            })
        }

        // Particles without mass last
        for c in &self.constructions {
            if let Construction::Galaxy {
                center_pos,
                center_vel,
                center_mass,
                amount,
                normal,
            } = c
            {
                galaxygen::generate_galaxy(
                    &mut particles,
                    *amount,
                    self.safety,
                    (*center_pos).into(),
                    (*center_vel).into(),
                    *center_mass,
                    (*normal).into(),
                );
            }
        }

        particles
    }
}
