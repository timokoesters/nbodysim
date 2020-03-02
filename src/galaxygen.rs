//! This module can generate spiral galaxies based on some parameters.

use {
    crate::Particle,
    cgmath::{
        prelude::*,
        {Point3, Vector3},
    },
    rand::prelude::*,
    std::f32::consts::PI,
};

const G: f64 = 6.67408E-11;
const ARMS: u32 = 2;

/// Fill the particles vector with many stars of a spiral galaxy.
pub fn generate_galaxy(
    particles: &mut Vec<Particle>,
    amount: u32,
    safety: f64,
    center_pos: Point3<f32>,
    center_vel: Vector3<f32>,
    center_mass: f64,
    mut normal: Vector3<f32>,
) {
    // Helpers
    normal = normal.normalize();
    let tangent = normal.cross(Vector3::new(-normal.z, normal.x, normal.y));
    let bitangent = normal.cross(tangent);

    // Generate center of the galaxy
    for _ in 0..amount / 5 {
        let radius = 5E9
            + (rand_distr::Normal::<f32>::new(0.0, 1E11)
                .unwrap()
                .sample(&mut thread_rng()))
            .abs();
        let angle = thread_rng().gen::<f32>() * 2.0 * PI;

        let diff = tangent * angle.sin() + bitangent * angle.cos();

        let fly_direction = diff.cross(normal).normalize();

        let pos = center_pos + diff * radius;

        let mass = 0E30;
        let density = 1.408;

        // Fg = Fr
        // G * m1 * m2 / (r^2 + C) = m1 * v^2 / r
        // sqrt(G * m2 * r / (r^2 + C)) = v
        let speed = (G * center_mass * radius as f64 / (radius as f64 * radius as f64 + safety))
            .sqrt() as f32;
        let vel = center_vel + fly_direction * speed;

        particles.push(Particle::new(pos, vel, mass, density));
    }

    // Generate spiral arms of the galaxy
    for _ in 0..amount / 5 * 4 {
        // Choose arm
        let arm = rand_distr::Uniform::from(0..ARMS).sample(&mut thread_rng());

        let radius = 5E9
            + (rand_distr::Normal::<f32>::new(0.0, 1E11)
                .unwrap()
                .sample(&mut thread_rng()))
            .abs();

        let angle = arm as f32 / ARMS as f32 * 2.0 * PI - radius * 1E-11
            + rand_distr::Normal::new(0.0, PI / 16.0)
                .unwrap()
                .sample(&mut thread_rng());

        let diff = tangent * angle.sin() + bitangent * angle.cos();

        let fly_direction = diff.cross(normal).normalize();

        let pos = center_pos + diff * radius;

        let mass = 0E30;
        let density = 1.408;

        // Fg = Fg
        // G * m1 * m2 / (r^2 + C) = m1 * v^2 / r
        // sqrt(G * m2 * r / (r^2 + C)) = v
        let speed = (G * center_mass * radius as f64 / (radius as f64 * radius as f64 + safety))
            .sqrt() as f32;
        let vel = center_vel + fly_direction * speed;

        particles.push(Particle::new(pos, vel, mass, density));
    }
}
