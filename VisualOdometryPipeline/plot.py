import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='Evaluate trajectory.')

parser.add_argument('files', type=str, help='files to evaluate', nargs='*')
parser.add_argument('--title', type=str, help='title of plot', default="")


if __name__ == "__main__":
    config = parser.parse_args()
    poses = np.loadtxt(config.files[0],delimiter=",")

    scale = 1
    if(len(config.files) > 1):
        ref = np.loadtxt(config.files[1])
        ref_traj = ref[:,3:12:4]
        scale = np.linalg.norm(ref_traj[0,:] - ref_traj[1,:])    
       
    fig,ax = plt.subplots()
    ax.plot(scale * poses[:,3],scale * poses[:,11],label="estimate")
    if(len(config.files) > 1):
        ax.plot(ref[:,3],ref[:,11],label="groudtruth")
    ax.axis('equal')
    ax.set_title(config.title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    plt.savefig("trajectory.png")