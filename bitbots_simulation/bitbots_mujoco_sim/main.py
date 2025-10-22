import mujoco
from mujoco import viewer


class Scene:
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

    def launch(self):
        with viewer.launch_passive(self.model, self.data) as view:
            while view.is_running() and self.data.time < 10000.0:
                mujoco.mj_step(self.model, self.data)


field = Scene("xml/adult_field.xml")
# goal = Scene("xml/goal.xml")

field.launch()
