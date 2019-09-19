use rand::prelude::*;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

const G: f32 = 6.67408E-11;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    mass: f32,
    _p: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Globals {
    particles: u32,
    zoom: f32,
    delta: f32,
    _p: f32,
}

impl Particle {
    fn new(pos: [f32; 2], vel: [f32; 2], mass: f32) -> Self {
        Self {
            pos: pos,
            vel: vel,
            mass,
            _p: 0.0,
        }
    }
    fn random(mass_distribution: impl Distribution<f32>) -> Self {
        Self {
            pos: [
                if thread_rng().gen() { -3E9 } else { 3E9 } + randc() * 1E9,
                randc() * 3E8,
            ],
            vel: [randc() * 1E3, randc() * 7E5],
            mass: (thread_rng().sample(mass_distribution) + 1.0) * 2E26,
            _p: 0.0,
        }
    }
}

// Returns a random coordinate from -1 to 1
fn randc() -> f32 {
    (thread_rng().gen::<f32>() - 0.5) * 2.0
}

fn generate_galaxy(particles: &mut Vec<Particle>, amount: u32, center: &Particle) {
    for i in 0..amount {
        let dp = randc() * 5E9;

        let mut pos = center.pos;
        pos[0] += dp;

        let mass = 0.0;

        // Fg = Fg
        // G * m1 * m2 / r^2 = m1 * v^2 / r
        // sqrt(G * m2 / r) = v

        let vel = [0.0, (G * center.mass / dp).sqrt()];
        particles.push(Particle::new(pos, vel, mass));
    }
}

fn main() {
    let mass_distribution = rand_distr::Exp::new(0.4).unwrap();

    let mut particles = Vec::new();

    let center = Particle::new([-4E9, 0.0], [0.0, 0.0], 1E30);
    generate_galaxy(&mut particles, 300, &center);
    particles.push(center);

    let center2 = Particle::new([4E9, 0.0], [0.0, 0.0], 1E30);
    generate_galaxy(&mut particles, 300, &center2);
    particles.push(center2);

    let globals = Globals {
        particles: particles.len() as u32,
        zoom: 1E-10,
        delta: 60.0,
        _p: 0.0,
    };

    run(globals, particles);
}

fn run(globals: Globals, particles: Vec<Particle>) {
    let particles_size = (particles.len() * std::mem::size_of::<Particle>()) as u64;

    let event_loop = EventLoop::new();

    let size = (1080, 1080);

    #[cfg(not(feature = "gl"))]
    let (window, instance, size, surface) = {
        use raw_window_handle::HasRawWindowHandle as _;

        let window = winit::window::Window::new(&event_loop).unwrap();

        let instance = wgpu::Instance::new();
        let surface = instance.create_surface(window.raw_window_handle());

        (window, instance, size, surface)
    };

    #[cfg(feature = "gl")]
    let (window, instance, size, surface) = {
        let wb = winit::WindowBuilder::new();
        let cb = wgpu::glutin::ContextBuilder::new().with_vsync(true);
        let context = cb.build_windowed(wb, &event_loop).unwrap();

        let (context, window) = unsafe { context.make_current().unwrap().split() };

        let instance = wgpu::Instance::new(context);
        let surface = instance.get_surface();

        (window, instance, size, surface)
    };

    window.set_resizable(false);

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
    });

    let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

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
        usage: wgpu::BufferUsage::MAP_READ
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    // Create a new buffer
    let current_buffer = device
        .create_buffer_mapped(
            particles.len(),
            wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
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

    // Describe the rendering process
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

    let mut swap_chain = device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Vsync,
        },
    );

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            event::Event::EventsCleared => {
                let frame = swap_chain.get_next_texture();
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
                encoder.copy_buffer_to_buffer(&current_buffer, 0, &old_buffer, 0, particles_size);
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.view,
                            resolve_target: None,
                            load_op: wgpu::LoadOp::Clear,
                            store_op: wgpu::StoreOp::Store,
                            clear_color: wgpu::Color::BLACK,
                        }],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..particles.len() as u32, 0..1);
                }

                device.get_queue().submit(&[encoder.finish()]);
            }
            _ => (),
        }
    });
}
