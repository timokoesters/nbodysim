mod config;
mod galaxygen;
mod render;

// All lengths in light seconds, all velocities in speed of light, all times in seconds
use cgmath::prelude::*;
use cgmath::{Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3};
use config::{Config, Construction};
use rand::prelude::*;
use ron::de::from_reader;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::fs::File;
use std::time::Instant;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

const SOLAR_MASS: f64 = 1.98847E30;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Particle {
    pos: Point3<f32>, // 4, 8, 12
    radius: f32,      // 16

    vel: Vector3<f32>, // 4, 8, 12
    _p: f32,           // 16

    mass: f64,     // 4, 8
    _p2: [f32; 2], // 12, 16
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Globals {
    matrix: Matrix4<f32>,    // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    camera_pos: Point3<f32>, // 16, 17, 18
    particles: u32,          // 19
    safety: f64,             // 20, 21
    delta: f32,              // 22
    _p: f32,                 // 23
}

impl Particle {
    fn new(pos: Point3<f32>, vel: Vector3<f32>, mass: f64, density: f64) -> Self {
        Self {
            pos,
            // V = 4/3*pi*r^3
            // V = m/ d
            // 4/3*pi*r^3 = m / d
            // r^3 = 3*m / (4*d*pi)
            // r = cbrt(3*m / (4*d*pi))
            radius: (3.0 * mass / (4.0 * density * PI as f64)).cbrt() as f32,
            vel,
            mass,
            _p: 0.0,
            _p2: [0.0; 2],
        }
    }
}

fn main() {
    let input_path = format!("{}/examples/two-galaxies.ron", env!("CARGO_MANIFEST_DIR"));
    let f = File::open(&input_path).expect("Failed opening file");
    let config: Config = from_reader(f).expect("Failed to load config!");
    let particles = config.construct_particles();

    let globals = Globals {
        matrix: Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.0)),
        camera_pos: config.camera_pos.into(),
        particles: particles.len() as u32,
        safety: config.safety,
        delta: 0.0,
        _p: 0.0,
    };

    render::run(globals, particles);
}
