# ShapeFromShading
     Your task is to develop a system that can predict surface normals given a single image. In this competition, each image is a synthetic image showing one object under simple lighting. We also provide a mask (a binary png image) for each image. A pixel in the mask is white if it is occupied by the object. You are free to use the mask in any way you like (e.g. as an additional input to your system). 
     The predicted surface normals are stored in png files. You need to predict a 3D vector at each pixel. That is, for each input image, your output is a color image with three channels that represents per-pixel surface normals.  
     We provide 20K images as the training set and 2K images as the test set. The ground truth for the training set is provided. Your task is to generate predictions for the 2K images in the test set. 
