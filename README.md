# DoodleGan
This project tries to create new doodles using GAN. This project uses google's quick draw data set
The dataset was too big put it in here. Here's the link. Go download it : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

Download any doodle data set you want and you can merge it if you want. This repo uses just single aircraft doodle drawn by millions of people and tries to create new aircraft doodle.

You can use this model even if you're using multiple datasets. All you need to do is merge them into one single data set and feed it in. That's it.
Go have fun!!

P.S: This repo doesn't contain the trained model. (Because it was too big to upload)
One thing to notice here is that the GAN model here uses around 500 epochs and even after that it doens't give good results. 
Feel free to play around with that parameter. You might need to put it somewhere around 2000 to 5000 epochs. It would take around 10-15 hours with GPU enabled training. 
