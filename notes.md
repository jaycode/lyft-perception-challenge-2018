username: MacDriver

First submission:

2018-05-17-2057
FC DenseNet 56
300 epochs
1.111 FPS
Car F score: 0.686 | Car Precision: 0.712 | Car Recall: 0.680 | Road F score: 0.977 | 
Road Precision: 0.977 | Road Recall: 0.979 | Averaged F score: 0.832

MobileUNet equivalent:

2018-05-21-1306
MobileUNet
300 epochs
1.363 FPS
Car F score: 0.666 | Car Precision: 0.671 | Car Recall: 0.665 | Road F score: 0.977 |
Road Precision: 0.977 | Road Recall: 0.976 | Averaged F score: 0.822

Both of the above models had an incorrect last layer that caused them to run very slow. 
The results below have corrected that issue.

Todo:

Compare 100 epochs no augment vs 100 epoch with augment.

Score of 100 epochs no augment:

2018-05-21-1714
MobileUNet
100 epochs
1.363 FPS
Car F score: 0.700 | Car Precision: 0.599 | Car Recall: 0.730 | Road F score: 0.976 |
Road Precision: 0.976 | Road Recall: 0.977 | Averaged F score: 0.838

By adding

```
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
```

The FPS went up to 2.0!


Score of 100 epochs with augment:

2018-05-21-2128
MobileUNet
100 epochs
1.578 FPS
Car F score: 0.577 | Car Precision: 0.428 | Car Recall: 0.631 | Road F score: 0.769 |
Road Precision: 0.730 | Road Recall: 0.977 | Averaged F score: 0.673


Augmented result is lower, but I have a hunch it may have a larger potential.
We see that in the non-augmented's case, 100s epoch result was better than 300 epochs.

Score of 300 epochs with augment:
2018-05-22-1815
MobileUNet
300 epochs
1.363 FPS
Car F score: 0.627 | Car Precision: 0.645 | Car Recall: 0.622 | Road F score: 0.691 |
Road Precision: 0.643 | Road Recall: 0.982 | Averaged F score: 0.659

Okay, I was wrong. Augmentation does not seem to help. Probably due to the image crops?

One problem, however, is with the hood of the car. Somehow it was always classified as a
road. I'd like to see what would happen if we can load the previous non-augmented model,
and then continue the training with augmented data. Let's see if it will get the best
of two worlds (or not).

Let's load model 2018-05-22-0345 first then continue with augmented data of 100 epochs.
Howver, I do not know how to do this, somehow the loss seems to keep reverting back. So let's skip this step for now.

# Augment with only flip and brightness

This method seems to give us the best result so far:

2018-05-23-0655
MobileUNet
300 epochs
1.428 FPS
Car F score: 0.779 | Car Precision: 0.638 | Car Recall: 0.825 | Road F score: 0.978 |
Road Precision: 0.979 | Road Recall: 0.974 | Averaged F score: 0.879

I tried adding erosion and dilation with the hope of removing the noises from the segmentation, but it didn't work that well it seems:

1.875 FPS
Car F score: 0.775 | Car Precision: 0.648 | Car Recall: 0.814 | Road F score: 0.978 |
Road Precision: 0.979 | Road Recall: 0.974 | Averaged F score: 0.876

However, the removal of output image resize had made the algorithm to run faster.


# CARLA Simulator

Run the following lines in separate windows:

```
../CARLA/CarlaUE4.sh -windowed -ResX=800 -ResY=600 -carla-server
```

```
./carla_client.py --autopilot --images-to-disk
```

We added 71 * 300 = 21300 images with other settings similar as above.

Test 100 epochs with them.

Score of 100 epochs +training data:

2018-05-26-1309
MobileUNet
100 epochs
1.363 FPS
Car F score: 0.761 | Car Precision: 0.727 | Car Recall: 0.770 | Road F score: 0.979 |
Road Precision: 0.979 | Road Recall: 0.979 | Averaged F score: 0.870

Added dilation then erosion (kernel=11) resulted in the following score:

Car F score: 0.799 | Car Precision: 0.686 | Car Recall: 0.833 | Road F score: 0.979 |
Road Precision: 0.979 | Road Recall: 0.978 | Averaged F score: 0.889


10 epochs of the above:
2018-05-28-0047
MobileUNet
10 epochs
9.523 FPS
Car F score: 0.774 | Car Precision: 0.710 | Car Recall: 0.791 | Road F score: 0.974 |
Road Precision: 0.974 | Road Recall: 0.974 | Averaged F score: 0.874


# Segment only the borders

Next experiment: Let's try to create segmentation only for car borders and then fill the contour. I hope this will fix the issue when segmenting car windows.

NOPE sorry my mistake, we are supposed to NOT include the see-through part of the car.


# Added dilation + cv2 to read video

Since the score gives more importance for recall, I used only a dilation of 3x3px kernel.
This will make the result larger, hence getting more parts of the car (i.e. more recall),
at the cost of also taking irrelevant sections (i.e. less precision).

The result:

2018-05-26-1309
MobileUNet
100 epochs
10.204 FPS
Car F score: 0.815 | Car Precision: 0.695 | Car Recall: 0.851 | Road F score: 0.977 |
Road Precision: 0.978 | Road Recall: 0.974 | Averaged F score: 0.896


# Do we need rotation data?

I'd like to compare 10 epochs of augmented without rotation (2018-05-28-0047) vs with
rotation (2018-05-29-0801), both without dilation.

Without rotation:
2018-05-28-0047
10.638 FPS
Car F score: 0.741 | Car Precision: 0.768 | Car Recall: 0.735 | Road F score: 0.974 |
Road Precision: 0.974 | Road Recall: 0.975 | Averaged F score: 0.858

With rotation (5 degrees max):
2018-05-29-0801
10.638 FPS
Car F score: 0.786 | Car Precision: 0.730 | Car Recall: 0.802 | Road F score: 0.970 |
Road Precision: 0.967 | Road Recall: 0.979 | Averaged F score: 0.878

With added dilation just like in the previous section:

2018-05-29-0801
MobileUNet
10 epochs
10.638 FPS
Car F score: 0.809 | Car Precision: 0.622 | Car Recall: 0.875 | Road F score: 0.970 |
Road Precision: 0.967 | Road Recall: 0.979 | Averaged F score: 0.889

Looks like we need them, but only for cars.

# Add rotation data to saved model

All results are without morphology adjustments.

20 epochs:

Car F score: 0.813 | Car Precision: 0.753 | Car Recall: 0.829 | Road F score: 0.976 |
Road Precision: 0.976 | Road Recall: 0.979 | Averaged F score: 0.894

38 epochs:

Car F score: 0.805 | Car Precision: 0.772 | Car Recall: 0.813 | Road F score: 0.977 |
Road Precision: 0.976 | Road Recall: 0.979 | Averaged F score: 0.891

Trying out 512x512 px:

2 epochs
3.333 FPS
Car F score: 0.755 | Car Precision: 0.639 | Car Recall: 0.790 | Road F score: 0.970 |
Road Precision: 0.969 | Road Recall: 0.977 | Averaged F score: 0.862

Back to 256x256px since it was too slow.

I also set batch size to 4 for a faster convergence. The results below reflects this settings unless otherwise stated.

256x256px, batch size 4, 43 epochs:

Your program runs at 11.111 FPS

Car F score: 0.806 | Car Precision: 0.776 | Car Recall: 0.813 | Road F score: 0.977 |
Road Precision: 0.977 | Road Recall: 0.979 | Averaged F score: 0.891

Looks like the model overfits.

# Comparing various color channels

All of the results below use no morph transformation.

Gray, 5 epochs, batch of 4:

11.111 FPS

Car F score: 0.696 | Car Precision: 0.794 | Car Recall: 0.675 | Road F score: 0.972 |
Road Precision: 0.970 | Road Recall: 0.977 | Averaged F score: 0.834

HSV, 5 epochs, batch of 4:

10.989 FPS

Car F score: 0.734 | Car Precision: 0.494 | Car Recall: 0.834 | Road F score: 0.960 |
Road Precision: 0.957 | Road Recall: 0.972 | Averaged F score: 0.847

RGB, 5 epochs, batch of 4:

10.989 FPS

Car F score: 0.750 | Car Precision: 0.666 | Car Recall: 0.775 | Road F score: 0.965 |
Road Precision: 0.961 | Road Recall: 0.981 | Averaged F score: 0.858


# More training data with RGB

Added 12 more episodes.

RGB beats both methods, although it had a lower Car Recall. I then trained RGB with 20 epochs:
2018-06-03-0419-ckpt-e20
10.989 FPS
Car F score: 0.815 | Car Precision: 0.719 | Car Recall: 0.843 | Road F score: 0.972 |
Road Precision: 0.969 | Road Recall: 0.983 | Averaged F score: 0.894

Added 5 epochs with batch of 2 each:

2018-06-03-1517-ckpt-e5
11.111 FPS
Car F score: 0.820 | Car Precision: 0.718 | Car Recall: 0.850 | Road F score: 0.973 |
Road Precision: 0.971 | Road Recall: 0.983 | Averaged F score: 0.897


10 epochs:
2018-06-03-1517-ckpt-e10
11.494 FPS

Car F score: 0.820 | Car Precision: 0.718 | Car Recall: 0.850 | Road F score: 0.973 |
Road Precision: 0.971 | Road Recall: 0.983 | Averaged F score: 0.897


11 epochs:

2018-06-03-1517-ckpt-e11
10.989 FPS

Car F score: 0.816 | Car Precision: 0.746 | Car Recall: 0.835 | Road F score: 0.974 |
Road Precision: 0.972 | Road Recall: 0.984 | Averaged F score: 0.895


There was an increase by using > 0.0 for vehicle and > 0.99 for road i.e. the result was
adjusted to focus more on recall for vehicle and precision for road. The results from
this point will use this method instead.

I ran 5 epochs of batch of 1:

2018-06-03-2214-ckpt-e5

Car F score: 0.822 | Car Precision: 0.731 | Car Recall: 0.848 | Road F score: 0.976 |
Road Precision: 0.976 | Road Recall: 0.980 | Averaged F score: 0.899

And the result with the previous 0.5 screening:

Car F score: 0.812 | Car Precision: 0.763 | Car Recall: 0.825 | Road F score: 0.975 |
Road Precision: 0.974 | Road Recall: 0.982 | Averaged F score: 0.894

It looks like more training no longer improves the model.


# Trained only on big loss

I trained the network only on images with big losses (1064 images). Result was super bad as if the network was not trained at all.





python algo-8.py --load_model ./saved_models/2018-06-01-0553 --save_big_loss 0.03 --save_big_loss_epochs 0
