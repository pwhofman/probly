Core Concepts
=============

1. Understanding Uncertainty in Machine Learning
---------------------------------------------

This section explains what uncertainty means in machine learning, why it naturally 
arises in real-world problems, and why handling it correctly is essential for 
building trustworthy models. Probly provides tools to work with uncertainty in a 
structured and unified way.

1.1 What Is Uncertainty?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In standard machine-learning pipelines, a model outputs a **single prediction** —
a class label, a probability, or a regression value. However, this number does not 
tell us **how confident** the model is about that prediction.

In machine learning, *uncertainty* refers to the **degree of confidence** a model 
has in its outputs. There are two key types:

**• Epistemic Uncertainty**  
   Uncertainty caused by lack of knowledge.  
   The model has not seen enough examples or has never seen a similar input before  
   (e.g., a rare medical image, an unusual object on the road).  
   *This uncertainty can be reduced by more or better data.*

**• Aleatoric Uncertainty**  
   Uncertainty caused by noise in the data itself.  
   Measurements may be noisy, labels may be ambiguous, or images may be blurry.  
   *This uncertainty cannot simply be removed by collecting more data.*

Most classical ML models — like neural networks or random forests — **ignore both 
forms of uncertainty** and return only a single output, leading to overconfident 
predictions.

Probly addresses this by offering unified tools to represent and quantify both 
aleatoric and epistemic uncertainty across different methods.

