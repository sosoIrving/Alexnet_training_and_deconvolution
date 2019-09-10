There are three parts,part1 and part3 are mutually independent
### Part1: Build and train an alexnet by one GPU
dataset: Imagenet2012 (http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)
frame: Tensorflow 1.14
state: the last top 5 accuracy rate is 77%,the reason maybe is that I didn't use data enhancement.

### Part3: Deconvoluting an Alexnet(the net is trained by two GPUs)
frame: keras
state: the weights you can find in: https://github.com/heuritech/convnets-keras

### Part2: convert .ckpt to .h5
I tried but failed. This is my first project. I tried to convert .ckpt file to .h5 file,but find that I can't.So part1 and part3
is independent.


### References:
https://github.com/FHainzl/Visualizing_Understanding_CNN_Implementation
https://blog.csdn.net/gzroy/article/details/87652291
