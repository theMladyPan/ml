# ml
Machine learning test repo

## run ROCm docker container
```shell
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network host -v /home/$USER:/home/$USER --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G themladypan/bark_test_rocm 
```