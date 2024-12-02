import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("dark_background")


times1 = np.load('results_rl/timesteps_dinno_50.npy')
rewards1 = np.load('results_rl/avg_ep_rews_dinno_50.npy')
loss= np.load('results_rl/avg_loss50.npy')
# times1 = np.load('results_sonata/timesteps_sonata_0.npy')
# rewards1 = np.load('results_sonata/avg_ep_rews_sonata_0.npy')
# times2 = np.load('trained/timesteps_2.npy')
# rewards2 = np.load('trained/avg_ep_rews_2.npy')
# times3 = np.load('trained/timesteps_3.npy')
# rewards3 = np.load('trained/avg_ep_rews_3.npy')

times_dsgt = np.load('trained/timesteps_dsgt_13.npy')
rewards0 = np.load('trained/avg_ep_rews_dsgt_10.npy')
# rewards_arr = np.vstack((rewards1, rewards2, rewards3))
rewards_arr = np.vstack((rewards1, rewards1, rewards1))




# 
fig, ax0 = plt.subplots(figsize=(20, 8), tight_layout = True)
# ax0.plot(times1, np.mean(rewards_arr, axis=0), c="indianred",label="sonata")
# ax0.plot(times_dsgt, rewards0,  label="DSGT")
# ax0.fill_between(times1, np.amax(rewards_arr, axis=0), np.amin(rewards_arr, axis=0), color="indianred", alpha=0.5)

ax0.legend()

ax0.set_title("RL Training Over Time")
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Average Episode Reward")

plt.show()




