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

double length2(dvec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

void main() {
    int i = gl_VertexIndex;

    // Early return
    if(data[i].mass < 0) { 
        gl_PointSize = 0;
        return;
    }

    // Render
    gl_Position = matrix * vec4(data[i].pos - camera_pos, 1.0);

    gl_PointSize = float(clamp(data[i].mass * 4E-35, 1.5, 20));

    if(data[i].mass > 1E35) {
        fragColor = vec3(0.0, 0.0, 0.0);
    } else {
        if(i%1000==0) {
            fragColor = vec3(1.0, 0.0, 0.0);
            gl_PointSize = 4;
        }
        else {
            fragColor = vec3(0.1, 0.0, 0.5);
        }
    }
}
