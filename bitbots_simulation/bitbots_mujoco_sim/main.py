import mujoco
from mujoco import viewer

m = mujoco.MjModel.from_xml_path("robot.urdf")
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as v:
    print("Time\tPos X\tPos Y\tPos Z")
    while v.is_running() and d.time < 5000.0:
        print(f"{d.time:.2f}\t{d.qpos[0]:.4f}\t{d.qpos[1]:.4f}\t{d.qpos[2]:.4f}")
        mujoco.mj_step(m, d)