#version 450

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

const float G = 6.67408E-11;
const float MIN_DISTANCE2 = pow(1E7, 2);

struct Particle {
    vec3 pos; // 0, 1, 2
    vec3 vel; // 4, 5, 6
    float mass; // 7
};

layout(set = 0, binding = 0) uniform GlobalsBuffer {
    mat4 matrix;
    vec3 camera_pos;
    uint particles;
    float zoom;
    float delta;
};

layout(std430, set = 0, binding = 1) buffer DataOld {
    Particle data_old[];
};

layout(std430, set = 0, binding = 2) buffer DataCurrent {
    Particle data[];
};

float rand(vec3 co) {
    return (fract(sin(dot(co.xyz, vec3(32.3485, 11.8743, 50.463))) * 48510.7134) - 0.5) * 2.0;
}

void main() {
    int i = gl_VertexIndex;

    if(data_old[i].mass < 0) { 
        gl_PointSize = 0;
        return;
    }

    // Update
    data[i].pos += data_old[i].vel * delta;
    vec3 real_pos = data_old[i].pos * zoom;

    // Render
    gl_Position = matrix * vec4(real_pos.xyz - camera_pos, 1.0);

    if(data_old[i].mass > 0.0) {
        gl_PointSize = sqrt(data_old[i].mass * 2E-27);
    } else {
        gl_PointSize = 2;
    }

    // Respawn particles
    //if(real_pos.x < -1 || real_pos.x > 1 || real_pos.y < -1 || real_pos.y > 1) {
    //    data[i].pos = vec2(rand(data_old[i].pos) * 5E9, rand(data_old[i].pos*2.356) * 5E9);
    //    data[i].vel = vec2(rand(data_old[i].pos*1.235) * 2E5, rand(data_old[i].pos*5.283) * 2E5);
    //}

    vec3 temp = vec3(0.0, 0.0, 0.0);

    // Gravity
    for(int j = 0; j < particles; j++) {
        if(j == i || data_old[j].mass == 0) { continue; }
        vec3 diff = data_old[i].pos - data_old[j].pos;
        float d2 = pow(length(diff), 2);
        if(d2 < MIN_DISTANCE2) {
            if(data_old[j].mass > 0 && data_old[i].mass == 0) {
                data[i].mass = -1;
            }
            continue;
        }
        vec3 dir = normalize(diff);

        temp += dir * data_old[j].mass / d2;
    }

    data[i].vel -= temp * G * delta;
}
