# GWO-Metaheuristic

<p align="justify">    
Convolutional Neural Networks applies kernels to extract characteristics from images such as edges, textures and patterns, this information is perform by computing derivates from the forward propagation method which executes a weighted sum through the whole model, once the prediction is done the error takes notes of the expected and obtained result on a loss function also known as fitness taking place the gradient descend learning method which minimizes the error in each iteration. 
</p>

<p align="justify">
Gradient descend is a well known machine learning algorithm however has limitations on the complexity of the network called vanished gradient and does not find the optimial solutions for non convex functions or problems which have multiple or equal solutions being sensitive of the inicial weights.
</p>

<p align="justify">
Metaheuristic algorithms by approaching different biology or non-biology methods explore in a search space limited by upper and lower bound variables a possibly new solution, close, equal or better than gradient descend, these algorithms could be applied to adapt the network architecture, adjust learning rate, regularizes, weights, coefficients or other hyperparameters that have a huge number of combinations or values.
</p>

<p align="justify">
Taking notes of how Convolutional Neural Networks extract features before the result is selected by the dense layers the amount and quality of data could be compromise, by adding a metaheuristic optimizations gray wolf on the final extracted features is perform a dimensionality reduction choosing the most relevant data, the performance of the selected features are test on a support vector machine ensemble boosting model, this means the data flows from the input through the last convolution layer which is applied a metaheuristic algorithm and only trains with the remaining data a support vector machine, through epochs the selected features change according the fitness function considering the native loss function and the amount of features selections above the total.
</p>

# How to Install

<p align="justify">

In order to execute and replicated the results the project was virtualized through docker being necessary only to install the tools to manage the virtual envirioment, aslo a devcontainer was made for visual studio code requiring less commands and a more friendly IDE to test and play on.    

</p>

### 1.- Docker Desktop

<p align="justify">

Download and procced a to install <a href = "www.docker.com/"> docker desktop </a> from the oficial website, once done enable the WSL2 connectivity currently on "General" and click on "Use the WSL2 based engine" or similar.

</p>

### 2.- WSL2

<p align="justify">

WSL2 is the windows subsytem for Linux, this will allow to integrate and use the kernel of many linux ditributions without need of virtual machine and being native on windows, to open it you must do the following path:

</p>

- Enable "Virtual Machine Plataform" on windows features, could be found in "Turn Windows features on or off"
- Open windows porwershell as an admin, then write "wsl --install"
- Write and login on Ubuntu distribucion a new user, then write "sudo apt-get update" to download all updates.

More information on <a href = "https://learn.microsoft.com/en-us/windows/wsl/install"> Microsoft </a>  oficial website.

## 3.- Nvidia Container Toolkit

<p align="justify">

Nvidia Container Toolkit allows the containers made by Docker the use of all Graphic Cards naturally this gives the support for tensorflow to utilize a GPU on training models, however is also need it to use nvidia cuda plataform to write parallel code and run it on the graphic card, this feature is applied with pycuda to makes faster the explotation process of the metaheuristic algorithm.

</p>

- Copy the following comand to download and install the files and restart the system.

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

More information on <a href = "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"> Nvidia </a>  oficial website.

## 4.- Verify Docker Instalation

- Go to the ubuntu distribucion download on WSL2 and type "docker --version" if everything is allright you should see the currently version of docker you got on your system.
- Run the next docker image "docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi" if you see your GPU, Drivers and Cuda version everything was installed correct.

## 5.- Run Container (Choose Envyroment)

### Ubuntu Terminal

Now you have two options to run the code one being run it on the native WSL2 ubuntu terminal which is allredy install for run the next command and wait around 30 minutos to the docker image to been fully download.

```
docker run -it --rm --gpus all --name GWOMetaheuristic pathinker/tensorflow-gpu-pycuda:2.18.0
```

The following meaning of the arguments send are the next:

- **it:** Allows and interactive terminal, it will allow to show feedback and logs from the terminal.
- **rm:** Removes the container once is close.
- **gpus all:** Gives access to the container all the GPUs on your device using Docker Container Toolkit
- **name:** Names the container.

For now you will need to write python3 and the complete route of files to execute them and other commands for editing, viewing the files, however all codes will be full oprational.

> [!CAUTION]
> Note: All changes made once the program is running will be loost due the missing of a volume that shares the data among the virtual envirioment and the host device.

### Visual Studio Code

To run it on Visual Studio Code you will need to search for extensions and type "Dev Containers" from Microsoft wait until is fully operational and then do the next shortcut "Ctrl + Shift + P", write and click on rebuild and open container. Then you will need to wait around the same time of the ubuntu terminal setup.

Using the devcontianer on Visual Studio Code will provide a few extensions to enable python debugging, fast code runner buttoms, access to your git SSH keys to clone and modify the repository, zsh terminal, data persistant between changes and not opening the WSL2 or Ubuntu terminal.

> [!IMPORTANT]
> Docker engine must be running everything you want to reopen the proyect in both cases.