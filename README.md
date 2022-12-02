# General Purpose Motion Detector

This is a general purpose motion detection plugin that incorporates various online moving object detectors, which can be paired with object trackers.

### Object Detectors:
* Background Subtraction (naive, Gaussian Mixture, and K-Nearest Neighbors methods)
	* [[reference paper]](https://www.sciencedirect.com/science/article/abs/pii/S0167865505003521)
* Farnebäck's Dense Optical Flow Method (recommended detector):
	* [[reference paper]](https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)

### Object Trackers:
* Naive Exponential Moving Average Tracker

## Dependencies

To run/test this plugin in a local Python3 environment, the required dependencies are:
```
numpy
opencv-contrib-python
```

You can install each dependency by running ``pip3 install <dependency>``

## funding
[NSF 1935984](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984)

## collaborators
Bhupendra Raut, Dario Dematties Reyes, Joseph Swantek, Neal Conrad, Nicola Ferrier, Pete Beckman, Raj Sankaran, Robert Jackson, Scott Collis, Sean Shahkarami, Sergey Shemyakin, Wolfgang Gerlach, Yongho kim
