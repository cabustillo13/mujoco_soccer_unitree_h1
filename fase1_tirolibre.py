import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np

def main():
    print("Cargando la escena oficial del Unitree Go2 desde Menagerie...")
    go2_handle = mm.get('unitree_go2')
    scene_path = str(go2_handle.path() / 'scene.xml')

    # 1. Cargar el MjSpec del scene.xml oficial del Go2
    spec = mujoco.MjSpec.from_file(scene_path)

    # 2. Agregar la pelota de fútbol al mundo
    ball_body = spec.worldbody.add_body(name="ball", pos=[0.8, 0.0, 0.15])
    ball_body.add_freejoint(name="ball_freejoint")
    ball_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.11],
        mass=0.43,
        rgba=[0.9, 0.9, 0.9, 1.0],
        friction=[0.6, 0.005, 0.0001],
        solref=[0.002, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2]
    )

    # 3. Compilar el modelo
    model = spec.compile()
    data = mujoco.MjData(model)

    # 4. Aplicar el keyframe inicial y guardar la postura de referencia erguida
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    
    mujoco.mj_forward(model, data)
    
    # Guardamos los ángulos articulares ideales del keyframe para usarlos como referencia del controlador
    target_qpos = np.copy(data.qpos)

    print("Abriendo visor interactivo con control de postura activo...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -25.0
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0.4, 0.0, 0.2]

        while viewer.is_running():
            
            # Controlador Proporcional (P) simple en los actuadores para mantener las patas firmes
            for act_id in range(model.nu):
                trnid = model.actuator_trnid[act_id, 0]
                jnt_qposadr = model.jnt_qposadr[trnid]
                
                q_current = data.qpos[jnt_qposadr]
                q_target = target_qpos[jnt_qposadr]
                
                # Ganancia de rigidez para los motores del Go2
                kp = 100.0
                torque = kp * (q_target - q_current)
                data.ctrl[act_id] = torque

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()