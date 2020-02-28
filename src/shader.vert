#version 450

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

layout(location = 0) out vec3 fragColor;

struct Particle {
    vec3 pos; // 0, 1, 2
    float radius; // 7
    vec3 vel; // 4, 5, 6
    double mass; // 7, 8
};

layout(set = 0, binding = 0) uniform GlobalsBuffer {
    mat4 matrix;
    vec3 camera_pos;
    uint particles;
    float delta;
};

layout(std430, set = 0, binding = 2) buffer DataCurrent {
    Particle data[];
};

void main() {
    int i = gl_VertexIndex;

    // Early return
    if(data[i].mass < 0) { 
        gl_PointSize = 0;
        return;
    }

    // Render
    gl_Position = matrix * vec4(data[i].pos, 1.0);

    if (data[i].mass > 0) {
        gl_PointSize = clamp(30 * 1E11 / gl_Position.z, 1, 20);
    } else {
        gl_PointSize = clamp(1 * 1E11 / gl_Position.z, 1, 5);
    }

    if(data[i].mass > 1E33) {
        // Color objects with big mass black
        fragColor = vec3(0.0, 0.0, 0.0);
    } else {
        // Give different colors to half of all particles
        if(i < particles/2+1) {
            fragColor = vec3(0.722, 0.22, 0.231);
        }
        else {
            fragColor = vec3(0.345, 0.522, 0.635);
        }
    }
}
