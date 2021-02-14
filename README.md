# Image-processing-assignments
Following questions were part of the assignment for the digital image processing course. 
1. Histogram.py was written to find the histogram for the grayscale images.
2. Frequency lowpass.py was written to check the frequency domain ideal low pass filter properties

3. Filter the image characters.tif in the frequency domain using the Gaussian low pass filter given by
𝐻 (𝑢, 𝑣) =exp (−𝐷2(𝑢, 𝑣)/2𝐷02),
where all the terms are as explained in part b. For 𝐷0=100, compare the result with that of the ILPF. 
This was done using Gauss-lowpass.py

4. Use homomorphic filtering to enhance the contrast of the image PET_image.tif. Use the following
filter to perform the high pass filtering
𝐷 (𝑢, 𝑣) =(𝛾𝐻−𝛾𝐿) [1−exp (−𝐷2(𝑢, 𝑣)/2𝐷02)] + 𝛾𝐿,
where 𝛾𝐻, 𝛾𝐿 and 𝐷0 are the parameters that you need to adjust through experimentation. 
This was done using Homomorphic filtering.py

5.a. Mitigate the noise in the image noisy.tif by filtering it with a square averaging mask of
sizes 5,10 and 15. What do you notice with increasing mask size.

b. Use high boost filtering to sharpen the denoised image from part a. Choose the scaling
constant for the high pass component that minimizes the mean squared error between the
sharpened image and the image characters.tif.
This was done using Spatial domain filtering.py

6. Generate a 𝑀×𝑁 sinusoidal image sin(2𝜋𝑢0𝑚/𝑀+2𝜋𝑣0𝑛/𝑁) for 𝑀=𝑁=1001, 𝑢0=100
and 𝑣0=200 and compute its DFT. To visualize the DFT of an image take logarithm of the
magnitude spectrum. This was done using DFT visualizer.py

7. Image Deblurring: Deblur the images Blurred-LowNoise.png (Noise Standard Deviation (𝜎)=1),
Blurred-MedNoise.png (𝜎=5) and Blurred-HighNoise.png (𝜎=10) which have been blurred by the
kernel BlurKernel.mat using inverse filtering, constrained least squares filtering
