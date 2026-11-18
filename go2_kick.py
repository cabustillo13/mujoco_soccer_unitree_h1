"""
Unitree Go2 camina y patea una pelota — MuJoCo (cancha verde tipo ajedrez)

Ideas clave:
  * Los actuadores del Go2 son MOTORES DE TORQUE -> hace falta un controlador PD
    propio (torque = kp*(q_obj - q) - kd*qdot). Con ctrl = posicion no pasa nada.
  * La base NO se fija: el robot es libre y se mueve solo con sus patas.
  * Marcha "trot" a lazo abierto (patas en diagonal) + patada con la pata FR.
"""
import sys
import numpy as np
import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import imageio

# ----------------------------------------------------------------- parametros
BALL_RADIUS = 0.11
BALL_MASS   = 0.40
BALL_X      = 0.95      # distancia inicial de la pelota (m)
BALL_Y      = -0.14     # alineada con la pata delantera derecha (FR)

T_STAND   = 1.0         # s parado
T_WALK    = 3.2         # s caminando (hasta aca)
T_KICK    = 3.2         # inicio patada
T_END     = 6.5

GAIT_FREQ = 2.2         # Hz
STRIDE    = 0.28        # amplitud del muslo (rad)
LIFT      = 0.55        # cuanto se pliega la rodilla al levantar la pata (rad)

KP, KD = 70.0, 1.2
KP_KICK, KD_KICK = 220.0, 3.0   # pata que patea: mas rigida
LEAN = 0.18                    # desplazamiento lateral del cuerpo al patear (rad)

# Colores de la cancha
GRASS_1 = [0.16, 0.50, 0.17]
GRASS_2 = [0.24, 0.62, 0.22]

HOME = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}
LEGS = ["FL", "FR", "RL", "RR"]


def build_model():
    path = mm.get("unitree_go2").path() / "scene.xml"
    spec = mujoco.MjSpec.from_file(str(path))

    # --- cancha verde a cuadros (modifico la textura del piso oficial)
    for tex in spec.textures:
        if tex.name == "groundplane":
            tex.rgb1 = GRASS_1
            tex.rgb2 = GRASS_2
            tex.markrgb = [0.9, 0.9, 0.9]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.texrepeat = [2, 2]      # mas chico = cuadros mas grandes
            mat.reflectance = 0.05

    # --- pelota
    ball = spec.worldbody.add_body(name="ball", pos=[BALL_X, BALL_Y, BALL_RADIUS])
    ball.add_freejoint()
    ball.add_geom(
        name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0],
        mass=BALL_MASS, rgba=[1, 1, 1, 1], friction=[0.6, 0.005, 0.02], condim=6,   # condim=6 -> friccion de rodadura (la pelota frena sola)
    )
    return spec.compile()


def gait_targets(t, model, speed):
    """Objetivos de articulacion (12) para el trot en el instante t."""
    q = {}
    for leg in LEGS:
        diag = 0.0 if leg in ("FL", "RR") else 0.5       # pares diagonales
        p = (GAIT_FREQ * t + diag) % 1.0
        thigh = HOME["thigh"] + speed * STRIDE * np.cos(2 * np.pi * p)
        swing = np.sin(2 * np.pi * p) if p < 0.5 else 0.0  # solo en el aire
        calf = HOME["calf"] - speed * LIFT * swing
        q[leg] = (HOME["hip"], thigh, calf)
    return q


def kick_targets(tk):
    """Pata FR: 1) levanta y arma (rodilla plegada, pie atras)  2) golpe rapido  3) recupera."""
    lift_t, hit_t, rec_t = 0.30, 0.14, 0.50
    if tk < lift_t:                                   # levantar la pata del suelo
        a = tk / lift_t
        a = a * a * (3 - 2 * a)
        return (0.0, 0.9 + 0.35 * a, -1.8 - 0.75 * a)
    if tk < lift_t + hit_t:                           # golpe: la pierna se extiende hacia adelante
        a = (tk - lift_t) / hit_t
        return (0.0, 1.25 - 2.15 * a, -2.55 + 1.75 * a)
    a = min(1.0, (tk - lift_t - hit_t) / rec_t)       # recuperar la pose de pie
    a = a * a * (3 - 2 * a)
    return (0.0, -0.8 + 1.7 * a, -0.8 - 1.0 * a)


def main(headless=False, out="go2_soccer_demo.mp4"):
    model = build_model()
    data = mujoco.MjData(model)

    # Pose inicial = keyframe oficial "home"
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    # Mapas nombre -> indices
    jq  = {n: model.jnt_qposadr[model.joint(n).id] for n in
           [f"{l}_{j}_joint" for l in LEGS for j in ("hip", "thigh", "calf")]}
    jv  = {n: model.jnt_dofadr[model.joint(n).id] for n in jq}
    act = {f"{l}_{j}": model.actuator(f"{l}_{j}").id for l in LEGS for j in ("hip", "thigh", "calf")}

    def pd(leg, targets, kp=KP, kd=KD):
        for j, qd in zip(("hip", "thigh", "calf"), targets):
            jn = f"{leg}_{j}_joint"
            tau = kp * (qd - data.qpos[jq[jn]]) - kd * data.qvel[jv[jn]]
            data.ctrl[act[f"{leg}_{j}"]] = tau

    # Video
    fps, W, H = 30, 960, 544
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90, -18, 2.6
    frames, next_frame = [], 0.0
    ball_id = model.body("ball").id
    kicked_speed = 0.0

    def step_once():
        nonlocal kicked_speed
        t = data.time
        # velocidad de marcha: rampa suave y frenado antes de patear
        if t < T_STAND:
            speed = 0.0
        elif t < T_WALK:
            speed = min(1.0, (t - T_STAND) / 0.6) * min(1.0, (T_WALK - t) / 0.4 + 0.0)
        else:
            speed = 0.0
        q = gait_targets(t, model, speed) if t < T_KICK else gait_targets(0, model, 0)
        # Antes de patear: desplazo el cuerpo hacia la izquierda (patas de apoyo se abren
        # hacia la derecha) para que el centro de masa quede dentro del triangulo de apoyo.
        lean = LEAN * np.clip((t - (T_KICK - 0.3)) / 0.3, 0, 1) if t < T_KICK + 1.2 else \
               LEAN * np.clip(1 - (t - T_KICK - 1.2) / 0.5, 0, 1)
        for leg in LEGS:
            h, th, ca = q[leg]
            pd(leg, (h - lean, th, ca))
        if t >= T_KICK:
            pd("FR", kick_targets(t - T_KICK), kp=KP_KICK, kd=KD_KICK)
        mujoco.mj_step(model, data)
        kicked_speed = max(kicked_speed, np.linalg.norm(data.cvel[ball_id][3:]))

    look = np.array([0.5, -0.05, 0.15])

    def update_cam(c):
        """Sigue al robot y, tras la patada, se desplaza hacia la pelota (suavizado)."""
        bx = data.xpos[ball_id][0]
        target_x = data.qpos[0] + 0.5
        if data.time > T_KICK + 0.2:
            target_x = 0.5 * (data.qpos[0] + min(bx, data.qpos[0] + 3.5)) + 0.3
        look[0] += 0.08 * (target_x - look[0])
        c.lookat[:] = look
        c.distance = 2.6 + (0.9 if data.time > T_KICK else 0.0)

    def grab():
        nonlocal next_frame
        if data.time >= next_frame:
            update_cam(cam)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            next_frame += 1.0 / fps

    if headless:
        while data.time < T_END:
            step_once(); grab()
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 90, -18, 2.6
            while v.is_running() and data.time < T_END:
                step_once(); grab()
                update_cam(v.cam)
                v.sync()

    print(f"Pelota final: x={data.qpos[model.jnt_qposadr[model.body('ball').jntadr[0]]]:.2f} m | "
          f"vel. max pelota: {kicked_speed:.1f} m/s | altura base: {data.qpos[2]:.2f} m")
    imageio.mimsave(out, frames, fps=fps)
    print("Video guardado:", out)


if __name__ == "__main__":
    main(headless="--headless" in sys.argv)