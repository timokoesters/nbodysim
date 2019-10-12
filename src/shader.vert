#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

struct Particle {
    vec2 pos;
    vec2 vel;
};

layout(set = 0, binding = 0) buffer Data {
    Particle data[];
};

void main() {
    data[gl_VertexIndex].vel += 0.001 * (-data[gl_VertexIndex].pos);
    data[gl_VertexIndex].pos += data[gl_VertexIndex].vel;
    gl_Position = vec4(data[gl_VertexIndex].pos, 0.0, 1.0);
}
