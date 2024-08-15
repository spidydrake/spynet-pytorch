(c) Caner Hazirbas, Philipp Fischer, Vladimir Golkov, Alexey Dosovitskiy, 2015.

The "Flying Chairs" is a synthetic dataset with optical flow ground truth. It consists of 22872 image pairs and corresponding flow fields. Images show renderings of 3D chair models moving in front of random backgrounds from Flickr. Motions of both the chairs and the background are purely planar. 

We used this dataset for training ConvNets in the ICCV-2015 paper "FlowNet: Learning Optical Flow with Convolutional Networks" http://lmb.informatik.uni-freiburg.de//Publications/2015/DFIB15 .
The dataset is provided for research purposes only and without any warranty. 
Any commercial use is prohibited. If you use the dataset in your research, please cite the aforementioned paper (see bibtex below).

Images are in .ppm format, flow fields are in the .flo format. C++ and Matlab tools for reading .flo files are provided here: http://vision.middlebury.edu/flow/submit/ .

For help and bug reports please write to Alexey Dosovitskiy dosovits@cs.uni-freiburg.de

------

@InProceedings\{DFIB15,
  author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Hazirbas and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  month        = "Dec",
  year         = "2015",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2015/DFIB15"
}
