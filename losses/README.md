
All the custom losses we use, and their utility functions are defined here.

batch_rgb2ycbcr.py - convert batches of RGB data to the Y-Cb-Cr color space. Needed for MS-SSIM(Y) and MSE(Cb, Cr).  
gradient_loss.py - calculate the image gradient loss over a batch of images.  
ms_ssim.py - calculate the multiscale SSIM metric over a batch of images. Use level = 3 if the images are small. By default, it converts the image to grayscale before working on it.  
ycbcr_ms_ssim.py - calculate the multiscale SSIM metric for the Y channel over a batch of images.  
mef_ssim_color_loss.py - To calculate mef-ssim-c loss used in unsupervised static fusion as proposed here https://ece.uwaterloo.ca/~k29ma/papers/18_TCI_MEFOpt.pdf 
