# Style-Transfer

This project was my first big leap into the world of neural networks.  It was based off of the paper **Image Style Transfer Using Convolutional Neural Networks** which can be read here: 
http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf 

A VGG-19 neural network trained on object detection from the ImageNet database was used to extract style and content features from the input images.  Mean squared error was used to calcualte the loss between the input content image and the output.  The Gram Matrix is calculated to determine the style loss between the input and output.  After each iteration, the loss is minimized to generate an output that better matches the content and style of the inputs.  
## Results
