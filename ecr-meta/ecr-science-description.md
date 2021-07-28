# General Purpose Motion Detector

This is a general purpose motion detection plugin that incorporates various online moving object detectors, which can be paired with object trackers.

### Object Detectors:
* Background Subtraction (naive, Gaussian Mixture, and K-Nearest Neighbors methods)
	* [[reference paper]](https://www.sciencedirect.com/science/article/abs/pii/S0167865505003521)
* Farnebäck's Dense Optical Flow Method (recommended detector):
	* [[reference paper]](https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)

* TinyYOLOv2 (trained on the [PASCAL VOC dataset](https://www.kaggle.com/gopalbhattrai/pascal-voc-2012-dataset))
	* [[reference paper]](https://ieeexplore.ieee.org/document/7780460) 

### Object Trackers:
* Naive Exponential Moving Average Tracker

## Developer Notes
This plugin is still in development, and there are several features that would be great to add in the future. 
Some of these future changes may include:

* Replace tensornets TinyYOLO model with custom trained Tensorflow lite model

* Implement non-naive lightweight tracking models (e.g. [MOSSE filters](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf))

* Reformatting project structure (to better align with existing plugins)

