//! This module handles everything that has to do with the window. That includes opening a window,
//! parsing events and rendering. See shader.comp for the physics simulation algorithm.

use {
    crate::{Globals, Particle},
    cgmath::{prelude::*, Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3},
    std::{collections::HashSet, f32::consts::PI, time::Instant},
    wgpu::util::DeviceExt,
    winit::{
        event,
        event_loop::{ControlFlow, EventLoop},
    },
};

const TICKS_PER_FRAME: u32 = 3; // Number of simulation steps per redraw
const PARTICLES_PER_GROUP: u32 = 256; // REMEMBER TO CHANGE SHADER.COMP

fn build_matrix(pos: Point3<f32>, dir: Vector3<f32>, aspect: f32) -> Matrix4<f32> {
    Matrix4::from(PerspectiveFov {
        fovy: Rad(PI / 2.0),
        aspect,
        near: 1E8,
        far: 1E14,
    }) * Matrix4::look_to_rh(pos, dir, Vector3::new(0.0, 1.0, 0.0))
}

pub async fn run(mut globals: Globals, particles: Vec<Particle>) {
    // How many bytes do the particles need
    let particles_size = (particles.len() * std::mem::size_of::<Particle>()) as u64;

    let work_group_count = ((particles.len() as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

    let event_loop = EventLoop::new();

    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title("nbodysim");
    let window = builder.build(&event_loop).unwrap();

    // Pick a GPU
    let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
    let instance = wgpu::Instance::new(backend);
    let (mut size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(&window);
        (size, surface)
    };
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backend)
        .await
        .expect("No suitable GPU adapters found on the system!");

    println!("{:?}", adapter.get_info());

    // Try to grab mouse
    let _ = window.set_cursor_grab(true);

    window.set_cursor_visible(false);
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
        window.primary_monitor(),
    )));

    // Request access to that GPU
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device_descriptor"),
                features: wgpu::Features::SHADER_FLOAT64
                    | wgpu::Features::BUFFER_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    // Configure surface
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface
            .get_preferred_format(&adapter)
            .expect("surface compatible with adapter"),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Immediate,
    };
    surface.configure(&device, &config);

    // Load compute shader for the simulation
    let cs_module = device.create_shader_module(&wgpu::include_spirv!("shader.comp.spv"));

    // Load vertex shader to set calculate perspective, size and position of particles
    let vs_module = device.create_shader_module(&wgpu::include_spirv!("shader.vert.spv"));

    // Load fragment shader
    let fs_module = device.create_shader_module(&wgpu::include_spirv!("shader.frag.spv"));

    // Create globals buffer to give global information to the shader
    let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("globals"),
        contents: bytemuck::cast_slice(&[globals]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create buffer for the previous state of the particles
    let old_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("old_particles"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Create buffer for the current state of the particles
    let current_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("current_particles"),
        size: particles_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Texture to keep track of which particle is in front (for the camera)
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    });
    let mut depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Describe the buffers that will be available to the GPU
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind_group_layout"),
        entries: &[
            // Globals
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Old Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Current Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create the resources described by the bind_group_layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &bind_group_layout,
        entries: &[
            // Globals
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            // Old Particle data
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &old_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            // Current Particle data
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &current_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    // Combine all bind_group_layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    // Create render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "main",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList, // Draw vertices as blocky points
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            clamp_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState::IGNORE,
                back: wgpu::StencilFaceState::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
    });

    // Where is the camera looking at?
    let mut camera_dir = -globals.camera_pos.to_vec();
    camera_dir = camera_dir.normalize();
    globals.matrix = build_matrix(
        globals.camera_pos,
        camera_dir,
        size.width as f32 / size.height as f32,
    );

    // Speed of the camera
    let mut fly_speed = 1E10;

    // Which keys are currently held down?
    let mut pressed_keys = HashSet::new();

    // Vector that points to the right of the camera
    let mut right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();

    // Time of the last tick
    let mut last_tick = Instant::now();

    // Initial setup
    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Initialize current particle buffer
        encoder.copy_buffer_to_buffer(&old_buffer, 0, &current_buffer, 0, particles_size);

        queue.submit([encoder.finish()]);
    }

    // Start main loop
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
                            globals.delta = 1E0;
                        }
                        event::VirtualKeyCode::Key2 => {
                            globals.delta = 2E0;
                        }
                        event::VirtualKeyCode::Key3 => {
                            globals.delta = 4E0;
                        }
                        event::VirtualKeyCode::Key4 => {
                            globals.delta = 8E0;
                        }
                        event::VirtualKeyCode::Key5 => {
                            globals.delta = 16E0;
                        }
                        event::VirtualKeyCode::Key6 => {
                            globals.delta = 32E0;
                        }
                        event::VirtualKeyCode::F => {
                            let delta = last_tick.elapsed();
                            println!("delta: {:?}, fps: {:.2}", delta, 1.0 / delta.as_secs_f32());
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

                    fly_speed = fly_speed.min(1E13).max(1E9);
                }

                // Resize window
                event::WindowEvent::Resized(new_size) => {
                    size = new_size;
                    config.width = size.width;
                    config.height = size.height;
                    surface.configure(&device, &config);

                    // Reset depth texture
                    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("depth"),
                        size: wgpu::Extent3d {
                            width: size.width,
                            height: size.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    });
                    depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
                }
                _ => {}
            },

            // Simulate and redraw
            event::Event::RedrawRequested(_window_id) => {
                let delta = last_tick.elapsed();
                let dt = delta.as_secs_f32();
                last_tick = Instant::now();

                let frame = surface.get_current_frame().unwrap();
                let frame_view = frame
                    .output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                camera_dir.normalize();
                right = camera_dir.cross(Vector3::new(0.0, 1.0, 0.0));
                right = right.normalize();

                if pressed_keys.contains(&event::VirtualKeyCode::A) {
                    globals.camera_pos += -right * fly_speed * dt;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::D) {
                    globals.camera_pos += right * fly_speed * dt;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::W) {
                    globals.camera_pos += camera_dir * fly_speed * dt;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::S) {
                    globals.camera_pos += -camera_dir * fly_speed * dt;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::Space) {
                    globals.camera_pos.y -= fly_speed * dt;
                }

                if pressed_keys.contains(&event::VirtualKeyCode::LShift) {
                    globals.camera_pos.y += fly_speed * dt;
                }

                globals.matrix = build_matrix(
                    globals.camera_pos,
                    camera_dir,
                    size.width as f32 / size.height as f32,
                );

                // Create new globals buffer
                let new_globals_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("new_globals"),
                        contents: bytemuck::cast_slice(&[globals]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                    });

                // Upload the new globals buffer to the GPU
                encoder.copy_buffer_to_buffer(
                    &new_globals_buffer,
                    0,
                    &globals_buffer,
                    0,
                    std::mem::size_of::<Globals>() as u64,
                );

                // Compute the simulation a few times
                for _ in 0..TICKS_PER_FRAME {
                    encoder.copy_buffer_to_buffer(
                        &current_buffer,
                        0,
                        &old_buffer,
                        0,
                        particles_size,
                    );
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cpass"),
                    });
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch(work_group_count, 1, 1);
                }

                {
                    // Render the current state
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("rpass"),
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &frame_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.03,
                                    g: 0.03,
                                    b: 0.03,
                                    a: 1.0,
                                }),
                                store: false,
                            },
                        }],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: false,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(0),
                                store: false,
                            }),
                        }),
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..particles.len() as u32, 0..1);
                }

                queue.submit([encoder.finish()]);
            }

            // No more events in queue
            event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
