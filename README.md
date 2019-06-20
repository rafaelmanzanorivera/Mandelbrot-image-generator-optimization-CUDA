# Mandelbrot-image-generator-optimization-CUDA
I took existing Mandelbrot image generator from https://www-fourier.ujf-grenoble.fr/mobile/~parisse/info/makepng.cc and parallelized it using CUDA.

Results: 
  
  
On an 14177x14177 image:
  
   All program (including write png which is not parallelized):
       
       Serial time: 89,69" --> GPU time: 2,23" (speedUp = 40,42)
       
   Just Mandelbrot calculation part (compared against serial calculation)
   
       Serial time: 74,4371" --> GPU time: 0.0536" (speedUp = 1387,8) 
   
