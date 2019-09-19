#version 450

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

const float G = 6.67408E-11;
const float MIN_DISTANCE2 = pow(1E8, 2);

struct Particle {
    vec2 pos;
    vec2 vel;
    float mass;
};

layout(set = 0, binding = 0) uniform GlobalsBuffer {
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

float rand(vec2 co) {
    return (fract(sin(dot(co.xy, vec2(11.8743, 50.463))) * 48510.7134) - 0.5) * 2.0 ;
}

void main() {
    int i = gl_VertexIndex;

    // Update
    data[i].pos += data[i].vel * delta;
    vec2 real_pos = data_old[i].pos * zoom;
    gl_Position = vec4(real_pos, 0.0, 1.0);
    gl_PointSize = sqrt(data_old[i].mass * 2E-27);

    // Respawn particles
    if(real_pos.x < -1 || real_pos.x > 1 || real_pos.y < -1 || real_pos.y > 1) {
        data[i].pos = vec2(rand(data_old[i].pos) * 5E9, rand(data_old[i].pos*2.356) * 5E9);
        data[i].vel = vec2(rand(data_old[i].pos*1.235) * 2E5, rand(data_old[i].pos*5.283) * 2E5);
    }

    vec2 temp = vec2(0.0, 0.0);

    // Gravity
    for(int j = 0; j < particles; j++) {
        if(j == i || data_old[j].mass == 0) { continue; }
        vec2 diff = data_old[i].pos - data_old[j].pos;
        vec2 dir = normalize(diff);
        float d2 = pow(length(diff), 2);
        if(d2 < MIN_DISTANCE2) { continue; }

        temp += dir * data_old[j].mass / d2;
    }

    data[i].vel = data_old[i].vel - temp * G * delta;
}
