use cgmath::prelude::*;
use cgmath::{Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3};
use rand::prelude::*;
use std::collections::HashSet;
use std::f32::consts::PI;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

const G: f64 = 6.67408E-11;
const SOLAR_MASS: f64 = 1.98847E30;
const LIGHT_YEAR: f32 = 9.4607E15;
const LIGHT_SPEED: f32 = 299792458.0;

const PARTICLES_PER_GROUP: u32 = 8; // REMEMBER TO CHANGE SHADER.COMP

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Particle {
    pos: Vector3<f32>, // 4, 8, 12
    radius: f32,       // 16

    vel: Vector3<f32>, // 4, 8, 12
    _p: f32,           // 16

    mass: f64,     // 4, 8
    _p2: [f32; 2], // 12, 16
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Globals {
    matrix: Matrix4<f32>,    // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    camera_pos: Point3<f32>, // 16, 17, 18
    particles: u32,          // 19
    delta: f32,              // 20
    _p: [f32; 3],            // 21, 22, 23
}

impl Particle {
    fn new(pos: Vector3<f32>, vel: Vector3<f32>, mass: f64, density: f64) -> Self {
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
            _p: 0E30,
            _p2: [0E30, 0E30],
        }
    }
}

fn generate_galaxy(
    particles: &mut Vec<Particle>,
    amount: u32,
    center: &Particle,
    mut normal: Vector3<f32>,
) {
    normal = normal.normalize();
    let tangent = normal.cross(Vector3::new(-normal.z, normal.x, normal.y));
    let bitangent = normal.cross(tangent);

    for _ in 0..amount {
        let radius = 1E3 * LIGHT_YEAR + (thread_rng().gen::<f32>().powf(2.0)) * 50E3 * LIGHT_YEAR;
        let angle = thread_rng().gen::<f32>() * 2.0 * PI;

        let diff = tangent * angle.sin() + bitangent * angle.cos();

        let fly_direction = diff.cross(normal).normalize();

        let pos = center.pos + diff * radius;

        let mass = 0E30;
        let density = 1.408;

        // Fg = Fg
        // G * m1 * m2 / (r^2 + C) = m1 * v^2 / r
        // sqrt(G * m2 * r / (r^2 + C)) = v

        let speed = (G * center.mass * radius as f64 / (radius as f64 * radius as f64 + 1E41))
            .sqrt() as f32;
        let vel = center.vel + fly_direction * speed;

        particles.push(Particle::new(pos, vel, mass, density));
    }
}

fn main() {
    let mut particles = Vec::new();

    let center = Particle::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
        10E6 * SOLAR_MASS,
        1.0,
    );
    let center2 = Particle::new(
        Vector3::new(0.0, -3E4 * LIGHT_YEAR, -1E5 * LIGHT_YEAR),
        Vector3::new(0.0, 0.0, 2E-7 * LIGHT_SPEED),
        10E6 * SOLAR_MASS,
        1.0,
    );

    particles.push(center);
    particles.push(center2);

    generate_galaxy(
        &mut particles,
        100_000,
        &center,
        Vector3::new(1.0, 0.0, 0.5),
    );
    generate_galaxy(
        &mut particles,
        100_000,
        &center2,
        Vector3::new(1.0, 1.0, 0.0),
    );

    let globals = Globals {
        matrix: Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.0)),
        camera_pos: Point3::new(1.0, 0.0, 0.0),
        particles: particles.len() as u32,
        delta: 0.0,
        _p: [0.0; 3],
    };

    run(globals, particles);
}

fn build_matrix(pos: Point3<f32>, dir: Vector3<f32>, aspect: f32) -> Matrix4<f32> {
    Matrix4::from(PerspectiveFov {
        fovy: Rad(PI / 2.0),
        aspect,
        near: 0.01,
        far: 1E8 * LIGHT_YEAR,
    }) * Matrix4::look_at_dir(pos, dir, Vector3::new(0.0, 1.0, 0.0))
        * Matrix4::from_translation(pos.to_vec())
}

fn run(mut globals: Globals, particles: Vec<Particle>) {
    let particles_size = (particles.len() * std::mem::size_of::<Particle>()) as u64;
    let work_group_count = ((particles.len() as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

    let event_loop = EventLoop::new();

    #[cfg(not(feature = "gl"))]
    let (window, mut size, surface) = {
        let window = winit::window::Window::new(&event_loop).unwrap();

        let size = window.inner_size();

        let surface = wgpu::Surface::create(&window);

        (window, size, surface)
    };

    #[cfg(feature = "gl")]
    let (window, mut size, surface) = {
        let wb = winit::WindowBuilder::new();
        let cb = wgpu::glutin::ContextBuilder::new().with_vsync(true);
        let context = cb.build_windowed(wb, &event_loop).unwrap();

        let size = context
            .window()
            .get_inner_size()
            .unwrap()
            .to_physical(context.window().get_hidpi_factor());

        let (context, window) = unsafe { context.make_current().unwrap().split() };

        let surface = wgpu::Surface::create(&window);

        (window, size, surface)
    };

    // Try to grab mouse
    let _ = window.set_cursor_grab(true);

    window.set_cursor_visible(false);
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
        window.primary_monitor(),
    )));

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .unwrap();

    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    // Load compute shader
    let cs = include_str!("shader.comp");
    let cs_module = device.create_shader_module(
        &wgpu::read_spirv(glsl_to_spirv::compile(cs, glsl_to_spirv::ShaderType::Compute).unwrap())
            .unwrap(),
    );

    // Load vertex shader
    let vs = include_str!("shader.vert");
    let vs_module = device.create_shader_module(
        &wgpu::read_spirv(glsl_to_spirv::compile(vs, glsl_to_spirv::ShaderType::Vertex).unwrap())
            .unwrap(),
    );

    // Load fragment shader
    let fs = include_str!("shader.frag");
    let fs_module = device.create_shader_module(
        &wgpu::read_spirv(glsl_to_spirv::compile(fs, glsl_to_spirv::ShaderType::Fragment).unwrap())
            .unwrap(),
    );

    // Create a new buffer
    let globals_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST)
        .fill_from_slice(&[globals]);

    // Create a new buffer
    let old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: particles_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::STORAGE_READ
            | wgpu::BufferUsage::COPY_DST,
    });

    // Create a new buffer
    let current_buffer = device
        .create_buffer_mapped(
            particles.len(),
            wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::STORAGE_READ
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        )
        .fill_from_slice(&particles);

    // Describe the buffers that will be available to the GPU
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[
            // Globals
            wgpu::BindGroupLayoutBinding {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            },
            // Old Particle data
            wgpu::BindGroupLayoutBinding {
                binding: 1,
                visibility: wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                },
            },
            // Current Particle data
            wgpu::BindGroupLayoutBinding {
                binding: 2,
                visibility: wgpu::ShaderStage::COMPUTE | wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                },
            },
        ],
    });

    // Create the resources described by the bind_group_layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[
            // Globals
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &globals_buffer,
                    range: 0..std::mem::size_of::<Globals>() as u64,
                },
            },
            // Old Particle data
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &old_buffer,
                    range: 0..particles_size,
                },
            },
            // Current Particle data
            wgpu::Binding {
                binding: 2,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &current_buffer,
                    range: 0..particles_size,
                },
            },
        ],
    });

    // Combine all bind_group_layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    // Create render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::PointList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[],
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    });

    let mut swap_chain_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

    let mut camera_dir = -globals.camera_pos.to_vec();
    camera_dir = camera_dir.normalize();
    globals.matrix = build_matrix(
        globals.camera_pos,
        camera_dir,
        size.width as f32 / size.height as f32,
    );
    let mut fly_speed = LIGHT_YEAR;

    let mut pressed_keys = HashSet::new();

    let mut right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0));
    right = right.normalize();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            // Move mouse
            event::Event::DeviceEvent {
                event: event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                camera_dir = Quaternion::from_angle_y(Rad(-delta.0 as f32 / 300.0))
                    .rotate_vector(camera_dir);
                camera_dir = Quaternion::from_axis_angle(right, Rad(delta.1 as f32 / 300.0))
                    .rotate_vector(camera_dir);
            }

            event::Event::WindowEvent { event, .. } => match event {
                // Close window
                event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                // Keyboard input
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    match keycode {
                        // Exit
                        event::VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                        }
                        event::VirtualKeyCode::Key0 => {
                            globals.delta = 0.0;
                        }
                        event::VirtualKeyCode::Key1 => {
                            globals.delta = 1E13;
                        }
                        event::VirtualKeyCode::Key2 => {
                            globals.delta = 2E13;
                        }
                        event::VirtualKeyCode::Key3 => {
                            globals.delta = 4E13;
                        }
                        event::VirtualKeyCode::Key4 => {
                            globals.delta = 8E13;
                        }
                        event::VirtualKeyCode::Key5 => {
                            globals.delta = 16E13;
                        }
                        event::VirtualKeyCode::Key6 => {
                            globals.delta = 32E13;
                        }
                        event::VirtualKeyCode::Key7 => {
                            globals.delta = 64E13;
                        }
                        event::VirtualKeyCode::Key8 => {
                            globals.delta = 128E13;
                        }
                        event::VirtualKeyCode::Key9 => {
                            globals.delta = 256E13;
                        }
                        event::VirtualKeyCode::F11 => {
                            if window.fullscreen().is_some() {
                                window.set_fullscreen(None);
                            } else {
                                window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
                                    window.primary_monitor(),
                                )));
                            }
                        }
                        _ => {}
                    }
                    pressed_keys.insert(keycode);
                }

                // Release key
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state: event::ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    pressed_keys.remove(&keycode);
                }

                // Mouse scroll
                event::WindowEvent::MouseWheel { delta, .. } => {
                    fly_speed *= (1.0
                        + (match delta {
                            event::MouseScrollDelta::LineDelta(_, c) => c as f32 / 8.0,
                            event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 64.0,
                        }))
                    .min(4.0)
                    .max(0.25);

                    fly_speed = fly_speed.min(1E5 * LIGHT_YEAR).max(LIGHT_YEAR);
                }

                // Resize window
                event::WindowEvent::Resized(new_size) => {
                    size = new_size;
                    swap_chain_descriptor.width = new_size.width;
                    swap_chain_descriptor.height = new_size.height;
                    swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);
                }
                _ => {}
            },

            // Redraw
            event::Event::RedrawRequested(_window_id) => {
                let frame = swap_chain.get_next_texture();
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

                camera_dir.normalize();
                right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0));
                right = right.normalize();

                if pressed_keys.contains(&event::VirtualKeyCode::A) {
                    globals.camera_pos += -right * fly_speed;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::D) {
                    globals.camera_pos += right * fly_speed;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::W) {
                    globals.camera_pos += camera_dir * fly_speed;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::S) {
                    globals.camera_pos += -camera_dir * fly_speed;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::Space) {
                    globals.camera_pos.y -= fly_speed;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::LShift) {
                    globals.camera_pos.y += fly_speed;
                }

                globals.matrix = build_matrix(
                    globals.camera_pos,
                    camera_dir,
                    size.width as f32 / size.height as f32,
                );

                // Create new globals buffer
                let new_globals_buffer = device
                    .create_buffer_mapped(
                        1,
                        wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC,
                    )
                    .fill_from_slice(&[globals]);
                encoder.copy_buffer_to_buffer(
                    &new_globals_buffer,
                    0,
                    &globals_buffer,
                    0,
                    std::mem::size_of::<Globals>() as u64,
                );

                encoder.copy_buffer_to_buffer(&current_buffer, 0, &old_buffer, 0, particles_size);

                {
                    let mut cpass = encoder.begin_compute_pass();
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch(work_group_count, PARTICLES_PER_GROUP, 1);
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.view,
                            resolve_target: None,
                            load_op: wgpu::LoadOp::Clear,
                            store_op: wgpu::StoreOp::Store,
                            clear_color: wgpu::Color {
                                r: 0.03,
                                g: 0.03,
                                b: 0.03,
                                a: 1.0,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..particles.len() as u32, 0..1);
                }

                queue.submit(&[encoder.finish()]);
            }

            // No more events in queue
            event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
