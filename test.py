from simulation.functions import *
# endSim()
import numpy as np
# graspCube()
getLeftToPoint(False, 0.7, 0.0, 0.8, 0.05)
time.sleep(1)
sim_ret, Baxter_leftTip_handle = vrep.simxGetObjectHandle(clientID, 'Baxter_leftArm_tip', vrep.simx_opmode_blocking)
sim_ret, object_orientation = vrep.simxGetObjectOrientation(clientID, Baxter_leftTip_handle, -1, vrep.simx_opmode_blocking)
print("object_orientation:", object_orientation)
time.sleep(1)
vrep.simxSetObjectOrientation(clientID, Baxter_leftTip_handle, -1, (np.pi/2, np.pi/2, np.pi/2), vrep.simx_opmode_blocking)
sim_ret, object_orientation = vrep.simxGetObjectOrientation(clientID, Baxter_leftTip_handle, -1, vrep.simx_opmode_blocking)
print("sim_ret:", sim_ret)
print("object_orientation:", object_orientation)

# vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)

endSim()
