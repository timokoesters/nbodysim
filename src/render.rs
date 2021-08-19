//! This module handles everything that has to do with the window. That includes opening a window,
//! parsing events and rendering. See shader.comp for the physics simulation algorithm.

use {
    crate::{Globals, Particle},
    cgmath::{prelude::*, Matrix4, PerspectiveFov, Point3, Quaternion, Rad, Vector3},
    std::{collections::HashSet, f32::consts::PI, io::Cursor, time::Instant},
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

pub fn run(mut globals: Globals, particles: Vec<Particle>) {
    // How many bytes do the particles need
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

    // Pick a GPU
    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .unwrap();
    println!("{:?}", adapter.get_info());

    // Request access to that GPU
    let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    // Load compute shader for the simulation
    let cs = include_bytes!("shader.comp.spv");
    let cs_module = device.create_shader_module(&wgpu::read_spirv(Cursor::new(cs.iter())).unwrap());

    // Load vertex shader to set calculate perspective, size and position of particles
    let vs = include_bytes!("shader.vert.spv");
    let vs_module = device.create_shader_module(&wgpu::read_spirv(Cursor::new(vs.iter())).unwrap());

    // Load fragment shader
    let fs = include_bytes!("shader.frag.spv");
    let fs_module = device.create_shader_module(&wgpu::read_spirv(Cursor::new(fs.iter())).unwrap());

    // Create globals buffer to give global information to the shader
    let globals_buffer = device
        .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST)
        .fill_from_slice(&[globals]);

    // Create buffer for the previous state of the particles
    let old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: particles_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::STORAGE_READ
            | wgpu::BufferUsage::COPY_DST,
    });

    // Create buffer for the current state of the particles
    let current_buffer_initializer = device
        .create_buffer_mapped(particles.len(), wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&particles);

    let current_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: particles_size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_SRC
            | wgpu::BufferUsage::COPY_DST,
    });

    // Create swap chain to render images to
    let mut swap_chain_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Vsync,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

    // Texture to keep track of which particle is in front (for the camera)
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: swap_chain_descriptor.width,
            height: swap_chain_descriptor.height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
    });
    let mut depth_view = depth_texture.create_default_view();

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
                visibility: wgpu::ShaderStage::COMPUTE,
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
            cull_mode: wgpu::CullMode::Front,
            depth_bias: 2,
            depth_bias_slope_scale: 2.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::PointList, // Draw vertices as blocky points
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }),
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[],
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
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
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        // Initialize current particle buffer
        encoder.copy_buffer_to_buffer(
            &current_buffer_initializer,
            0,
            &current_buffer,
            0,
            particles_size,
        );

        queue.submit(&[encoder.finish()]);
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

                    // Reset swap chain, it's outdated
                    swap_chain_descriptor.width = new_size.width;
                    swap_chain_descriptor.height = new_size.height;
                    swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

                    // Reset depth texture
                    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                        size: wgpu::Extent3d {
                            width: swap_chain_descriptor.width,
                            height: swap_chain_descriptor.height,
                            depth: 1,
                        },
                        array_layer_count: 1,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Depth32Float,
                        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                    });
                    depth_view = depth_texture.create_default_view();
                }
                _ => {}
            },

            // Simulate and redraw
            event::Event::RedrawRequested(_window_id) => {
                let delta = last_tick.elapsed();
                let dt = delta.as_secs_f32();
                last_tick = Instant::now();

                let frame = swap_chain.get_next_texture();
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

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
                let new_globals_buffer = device
                    .create_buffer_mapped(
                        1,
                        wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_SRC,
                    )
                    .fill_from_slice(&[globals]);

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
                    let mut cpass = encoder.begin_compute_pass();
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch(work_group_count, 1, 1);
                }

                {
                    // Render the current state
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
                        depth_stencil_attachment: Some(
                            wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                attachment: &depth_view,
                                depth_load_op: wgpu::LoadOp::Clear,
                                depth_store_op: wgpu::StoreOp::Store,
                                clear_depth: 1.0,
                                stencil_load_op: wgpu::LoadOp::Clear,
                                stencil_store_op: wgpu::StoreOp::Store,
                                clear_stencil: 0,
                            },
                        ),
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
