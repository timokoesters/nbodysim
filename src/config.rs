use crate::{galaxygen, Particle};
use cgmath::prelude::*;
use cgmath::{Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3};
use rand::prelude::*;
use ron::de::from_reader;
use serde::Deserialize;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::time::Instant;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Deserialize, Clone, Debug)]
pub struct Config {
    pub camera_pos: [f32; 3],
    pub constructions: Vec<Construction>,
}

#[derive(Deserialize, Clone, Debug)]
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
    pub fn construct_particles(&self) -> Vec<Particle> {
        let mut particles = Vec::new();

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
