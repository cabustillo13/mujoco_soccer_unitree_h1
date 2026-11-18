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
BALL_X      = 1.27      # distancia inicial de la pelota (m)
BALL_Y      = -0.26     # alineada con la pata delantera derecha (FR)

T_STAND   = 1.0         # s parado
T_WALK    = 3.2         # s caminando (hasta aca)
T_KICK    = 3.2         # inicio patada
T_END     = 8.0

GAIT_FREQ = 2.2         # Hz
STRIDE    = 0.28        # amplitud del muslo (rad)
LIFT      = 0.55        # cuanto se pliega la rodilla al levantar la pata (rad)

KP, KD = 100.0, 1.5
KP_KICK, KD_KICK = 220.0, 3.0   # pata que patea: mas rigida
LEAN = 0.18                    # desplazamiento lateral del cuerpo al patear (rad)

# Arco (la pelota sale con un leve angulo a la derecha, asi que el arco esta sobre esa trayectoria)
GOAL_X, GOAL_Y = 5.0, -1.25     # posicion de la linea de gol (centro del arco)
GOAL_W, GOAL_H, GOAL_D = 2.0, 0.9, 0.7   # ancho, alto, profundidad (m)

# Colores de la cancha
GRASS_1 = [0.16, 0.50, 0.17]
GRASS_2 = [0.24, 0.62, 0.22]

# Postura de pie con las patas mas extendidas (el keyframe "home" es mas agachado: 0.9 / -1.8)
HOME = {"hip": 0.0, "thigh": 0.70, "calf": -1.40}
LEGS = ["FL", "FR", "RL", "RR"]


def add_goal(spec, x, y, w, h, d):
    """Arco: postes y travesano (colisionan) + redes translucidas."""
    wb, G = spec.worldbody, mujoco.mjtGeom
    r = 0.035
    white = [1, 1, 1, 1]

    def bar(p1, p2, name):
        wb.add_geom(name=name, type=G.mjGEOM_CAPSULE, size=[r, 0, 0],
                    fromto=[*p1, *p2], rgba=white)

    yl, yr = y - w / 2, y + w / 2
    # marco frontal
    bar((x, yl, 0), (x, yl, h), "post_left")
    bar((x, yr, 0), (x, yr, h), "post_right")
    bar((x, yl, h), (x, yr, h), "crossbar")
    # marco trasero y barras laterales
    hb = 0.6 * h
    bar((x + d, yl, 0), (x + d, yl, hb), "back_post_left")
    bar((x + d, yr, 0), (x + d, yr, hb), "back_post_right")
    bar((x + d, yl, hb), (x + d, yr, hb), "back_bar")
    bar((x, yl, h), (x + d, yl, hb), "side_bar_left")
    bar((x, yr, h), (x + d, yr, hb), "side_bar_right")

    # paneles de red invisibles: solo colisionan para que la pelota no se salga
    # (los hilos visuales se dibujan abajo; asi se evitan sombras raras de paneles translucidos)
    net = [1, 1, 1, 0]
    wb.add_geom(name="net_back", type=G.mjGEOM_BOX, size=[0.005, w / 2, hb / 2],
                pos=[x + d, y, hb / 2], rgba=net, solref=[0.03, 1], solimp=[0.5, 0.8, 0.01, 0.5, 2])
    for sign, nm in ((-1, "net_left"), (1, "net_right")):
        wb.add_geom(name=nm, type=G.mjGEOM_BOX, size=[d / 2, 0.005, hb / 2],
                    pos=[x + d / 2, y + sign * w / 2, hb / 2], rgba=net,
                    solref=[0.03, 1], solimp=[0.5, 0.8, 0.01, 0.5, 2])

    # hilos de la red (solo visuales)
    line = [1, 1, 1, 0.35]
    for yy in np.linspace(yl, yr, 11):
        wb.add_geom(type=G.mjGEOM_CAPSULE, size=[0.004, 0, 0], fromto=[x + d, yy, 0, x + d, yy, hb],
                    rgba=line, contype=0, conaffinity=0)
    for zz in np.linspace(0, hb, 6):
        wb.add_geom(type=G.mjGEOM_CAPSULE, size=[0.004, 0, 0], fromto=[x + d, yl, zz, x + d, yr, zz],
                    rgba=line, contype=0, conaffinity=0)
    for sign in (-1, 1):
        yy = y + sign * w / 2
        for xx in np.linspace(x, x + d, 5):
            zt = h + (hb - h) * (xx - x) / d
            wb.add_geom(type=G.mjGEOM_CAPSULE, size=[0.004, 0, 0], fromto=[xx, yy, 0, xx, yy, zt],
                        rgba=line, contype=0, conaffinity=0)
    # cinta de la linea de gol en el piso
    wb.add_geom(type=G.mjGEOM_BOX, size=[0.02, w / 2, 0.001], pos=[x, y, 0.001],
                rgba=[1, 1, 1, 1], contype=0, conaffinity=0)


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

    # --- arco
    add_goal(spec, GOAL_X, GOAL_Y, GOAL_W, GOAL_H, GOAL_D)

    # --- pelota
    ball = spec.worldbody.add_body(name="ball", pos=[BALL_X, BALL_Y, BALL_RADIUS])
    ball.add_freejoint()
    ball.add_geom(
        name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0],
        mass=BALL_MASS, rgba=[1, 1, 1, 1], friction=[0.6, 0.005, 0.004], condim=6,   # condim=6 -> friccion de rodadura (la pelota frena sola)
    )
    spec.stat.extent = 2.0                      # evita artefactos de sombras al agrandar la escena
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
        return (0.0, HOME['thigh'] + 0.55 * a, HOME['calf'] - 1.15 * a)
    if tk < lift_t + hit_t:                           # golpe: la pierna se extiende hacia adelante
        a = (tk - lift_t) / hit_t
        return (0.0, 1.25 - 2.15 * a, -2.55 + 1.75 * a)
    a = min(1.0, (tk - lift_t - hit_t) / rec_t)       # recuperar la pose de pie
    a = a * a * (3 - 2 * a)
    return (0.0, -0.9 + (HOME['thigh'] + 0.9) * a, -0.8 + (HOME['calf'] + 0.8) * a)


def draw_goal_text(img, since):
    """Cartel de ¡GOOOL! sobre el video (aparece con un pequeño efecto de zoom)."""
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    size = int(40 + 50 * min(1.0, since / 0.4))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    txt = "GOOOL!"
    x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=font)
    pos = ((pil.width - (x1 - x0)) // 2, 40)
    d.text((pos[0] + 3, pos[1] + 3), txt, font=font, fill=(0, 0, 0))
    d.text(pos, txt, font=font, fill=(255, 215, 0))
    return np.asarray(pil)


def main(headless=False, out="go2_soccer_demo.mp4"):
    model = build_model()
    model.vis.quality.shadowsize = 8192
    data = mujoco.MjData(model)

    # Pose inicial = keyframe oficial "home"
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    # Mapas nombre -> indices
    jq  = {n: model.jnt_qposadr[model.joint(n).id] for n in
           [f"{l}_{j}_joint" for l in LEGS for j in ("hip", "thigh", "calf")]}
    jv  = {n: model.jnt_dofadr[model.joint(n).id] for n in jq}
    act = {f"{l}_{j}": model.actuator(f"{l}_{j}").id for l in LEGS for j in ("hip", "thigh", "calf")}

    # Arrancar ya en la postura extendida
    for leg in LEGS:
        for j in ("hip", "thigh", "calf"):
            data.qpos[jq[f"{leg}_{j}_joint"]] = HOME[j]
    data.qpos[2] = 0.33
    mujoco.mj_forward(model, data)

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
    goal_time = None

    def step_once():
        nonlocal kicked_speed, goal_time
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
        bp = data.xpos[ball_id]
        if (goal_time is None and t > T_KICK and bp[0] > GOAL_X + 0.05
                and abs(bp[1] - GOAL_Y) < GOAL_W / 2 - 0.05 and bp[2] < GOAL_H):
            goal_time = data.time
            print(f"¡GOOOL! t={goal_time:.2f} s")
        kicked_speed = max(kicked_speed, np.linalg.norm(data.cvel[ball_id][3:]))

    look = np.array([0.5, -0.05, 0.15])

    def update_cam(c):
        """Sigue al robot y, tras la patada, se desplaza hacia la pelota (suavizado)."""
        bx = data.xpos[ball_id][0]
        target_x = data.qpos[0] + 0.5
        if data.time > T_KICK + 0.2:
            target_x = 0.5 * (data.qpos[0] + GOAL_X + GOAL_D) + 0.5   # encuadra robot y arco
        target_y = -0.05
        if data.time > T_KICK + 0.2:
            target_y = 0.5 * GOAL_Y
        look[1] += 0.05 * (target_y - look[1])
        look[0] += 0.08 * (target_x - look[0])
        c.lookat[:] = look
        c.distance = 2.6 + (1.7 if data.time > T_KICK else 0.0)

    def grab():
        nonlocal next_frame
        if data.time >= next_frame:
            update_cam(cam)
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            if goal_time is not None:
                img = draw_goal_text(img, data.time - goal_time)
            frames.append(img)
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