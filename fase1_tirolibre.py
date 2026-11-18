import mujoco
import mujoco.viewer
import mujoco_menagerie as mm

def apply_team_color(spec, color_rgba):
    """Recorre todas las geometrías de un spec y les asigna un color RGBA"""
    # Recorremos los cuerpos del spec y sus geoms para cambiarles el color
    for body in spec.worldbody.bodies:
        # Recursivamente o en sus geoms directas
        def set_geom_color(b):
            for geom in b.geoms:
                geom.rgba = color_rgba
            for child_b in b.bodies:
                set_geom_color(child_b)
        set_geom_color(body)

def build_scene():
    spec = mujoco.MjSpec()

    # Suelo y luz
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.1],
        rgba=[0.15, 0.25, 0.15, 1.0]
    )
    spec.worldbody.add_light(
        pos=[0, 0, 6],
        dir=[0, 0, -1],
        ambient=[0.5, 0.5, 0.5]
    )

    h1_handle = mm.get('unitree_h1')
    h1_scene_path = str(h1_handle.path() / 'scene.xml')

    # 1. Instancia Striker (Ej: Equipo Rojo / Borravino)
    striker_spec = mujoco.MjSpec.from_file(h1_scene_path)
    apply_team_color(striker_spec, [0.8, 0.1, 0.1, 1.0]) # Rojo
    frame_striker = spec.worldbody.add_frame(pos=[0.0, -2.5, 0.0])
    spec.attach(striker_spec, prefix="striker_", frame=frame_striker)

    # 2. Instancia Goalie (Ej: Equipo Azul)
    goalie_spec = mujoco.MjSpec.from_file(h1_scene_path)
    apply_team_color(goalie_spec, [0.1, 0.3, 0.9, 1.0]) # Azul
    frame_goalie = spec.worldbody.add_frame(pos=[0.0, 2.5, 0.0], quat=[0, 0, 1, 0])
    spec.attach(goalie_spec, prefix="goalie_", frame=frame_goalie)

    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data

def main():
    print("Construyendo escena multi-robot con Unitree H1 (Equipos identificables)...")
    model, data = build_scene()
    
    print(f"Total de cuerpos en la simulación (nbody): {model.nbody}")
    print(f"Total de articulaciones (njnt): {model.njnt}")

    print("Abriendo visor interactivo de MuJoCo...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()