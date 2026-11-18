import mujoco
import mujoco.viewer
import mujoco_menagerie as mm

def apply_team_color(spec, color_rgba):
    """Recorre todas las geometrías de un spec y les asigna un color RGBA"""
    for body in spec.worldbody.bodies:
        def set_geom_color(b):
            for geom in b.geoms:
                geom.rgba = color_rgba
            for child_b in b.bodies:
                set_geom_color(child_b)
        set_geom_color(body)

def build_scene():
    spec = mujoco.MjSpec()

    # 1. Suelo y luz global
    # Configuramos fricción en el piso para que la pelota ruede de forma realista
    floor_geom = spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[15, 15, 0.1],
        rgba=[0.15, 0.25, 0.15, 1.0],
        friction=[1.0, 0.005, 0.0001]
    )
    
    spec.worldbody.add_light(
        pos=[0, 0, 6],
        dir=[0, 0, -1],
        ambient=[0.5, 0.5, 0.5]
    )

    # 2. Agregar la Pelota de Fútbol de forma procedural
    ball_body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 0.15])
    # Le damos libertad de movimiento libre (flotante / free joint)
    ball_body.add_freejoint(name="ball_freejoint")
    
    # Geometría de la esfera (radio = 0.11m -> 11cm, masa estimada ~430g)
    ball_geom = ball_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11],
        mass=0.43,
        rgba=[0.9, 0.9, 0.9, 1.0], # Blanco brillante estilo pelota clásica
        friction=[0.6, 0.005, 0.0001],
        # Parámetros de elasticidad y disipación para el rebote del balón
        # solref=[tiempo_contacto, amortiguamiento], solimp=[impedancia]
        solref=[0.002, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2]
    )

    # 3. Cargar robots desde Menagerie
    h1_handle = mm.get('unitree_h1')
    h1_scene_path = str(h1_handle.path() / 'scene.xml')

    # Instancia Striker (Rojo) - Detrás de la pelota
    striker_spec = mujoco.MjSpec.from_file(h1_scene_path)
    apply_team_color(striker_spec, [0.8, 0.1, 0.1, 1.0])
    frame_striker = spec.worldbody.add_frame(pos=[0.0, -1.8, 0.0])
    spec.attach(striker_spec, prefix="striker_", frame=frame_striker)

    # Instancia Goalie (Azul) - Del otro lado custodiando
    goalie_spec = mujoco.MjSpec.from_file(h1_scene_path)
    apply_team_color(goalie_spec, [0.1, 0.3, 0.9, 1.0])
    frame_goalie = spec.worldbody.add_frame(pos=[0.0, 1.8, 0.0], quat=[0, 0, 1, 0])
    spec.attach(goalie_spec, prefix="goalie_", frame=frame_goalie)

    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data

def main():
    print("Construyendo escena de Fútbol 1v1 (Robots + Pelota)...")
    model, data = build_scene()

    print("Abriendo visor interactivo de MuJoCo...")
    print("Vas a ver la pelota blanca en el centro entre el robot rojo y el azul.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()