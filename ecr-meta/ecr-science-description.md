
# Science
This is a general purpose motion detection plugin that incorporates various online moving object detectors, that are Background subtraction [1] and Dense Optical Flow [2].

# AI@Edge
This application is intended for daylight hours. The daylight hours for a deployment is defined by the users, for example 7AM - 5PM. The application first collects data from a camera and then passes through the image processing methods either background extraction or dense optical flow to detect motion. When users run the plugin, they need to choose either background subtraction (bg_subtraction) or dense optical flow (dense_optflow) as an input argument. The output of this plugin is the detection of motion (True / False).

# Using the code
Output: Boolean motion detection report (0/1)  
Input: 5 second video  
Image resolution: any  
Inference time:  
Model loading time:  

# Arguments
   '--debug': Enable debug logs  
   '--input': Video input source  
   '--detector': The motion detector to use. In order from least to most computationally intensive, the options are: (1) bg_subtraction (2) dense_optflow (default = dense_optflow)  
   '--samples': Number of samples to publish (default = 1, run one time)  
   '--interval': iInterval between data publishes (in seconds) (default = 5)  

# Ontology
The code publishes measurements with topic ‘vision.motion_detected’

# References
[1] [Zivkovic, Zoran, and Ferdinand Van Der Heijden. "Efficient adaptive density estimation per image pixel for the task of background subtraction." Pattern recognition letters 27, no. 7 (2006): 773-780.](https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)  
[2] [Farnebäck, Gunnar. "Two-frame motion estimation based on polynomial expansion." In Scandinavian conference on Image analysis, pp. 363-370. Springer, Berlin, Heidelberg, 2003.](https://www.researchgate.net/publication/225138825_Two-Frame_Motion_Estimation_Based_on_Polynomial_Expansion)
