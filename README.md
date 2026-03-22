# TRAIL: Trajectory-Based Representation and Integration for Limiting Over-Smoothing
 
Link - https://link.springer.com/article/10.1007/s10994-026-07015-z

In order to operate this code, you must first install the packages listed in the requirement.txt. However, since everyone has a different computer environment, we strongly recommend using the docker container to build a code-driven environment.

For Docker Image, you can use the pyg image provided by Nvidia. The code in this paper was also experimented with that image. You can find the information about that image in the URL below.

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg?version=25.11-py3

For your information, the image tag used in this paper is nvcr.io/nvidia/pyg:24.05-py3 . Below is an example of a command to build a container.
Example) docker run -it -d --gpus all --name "container name" -v "your directory":"container directory" nvcr.io/nvidia/pyg:24.05-py3

+ But this image recently doesn't match with our code, we built the new docker image to run the code. (highly recommended)
Example) docker run -it -d --gpus all --name "container name" -v "your directory":"container directory" hamsss/gnn_env:latest
