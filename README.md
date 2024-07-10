# 🧪 OBIWAN 🧪

This is the code relative to the paper "OBIWAN: An Element-wise Scalable Feed-Forward Neural Network Potential" - [link](https://doi.org/10.1021/acs.jctc.4c00342).

Inspired by the effective ANAKIN method (2), we developed a feed-forward neural network potential whose architecture size doesn't scale with the number of considered species anymore, eventually allowing the user to re-utilise already learnt chemical knowledge without starting the training from scratch everytime.

Together with the training routine and all the other scripts used to produce the results in the article, you can find in [results/models](https://github.com/virtualmartire/OBIWAN/tree/master/results/models) the OBIWAN checkpoints necessary for the deployment of our model in the two explored scenarios and in [architectures/anakin.py](https://github.com/virtualmartire/OBIWAN/tree/master/architectures/anakin.py) an ANAKIN TensorFlow implementation for development purposes.

## Bibliography

1. S. Martire, S. Decherchi, and A. Cavalli, _OBIWAN: An Element-Wise Scalable Feed-Forward Neural Network Potential_, Journal of Chemical Theory and Computation, 2024, DOI: 10.1021/acs.jctc.4c00342
2. C. Devereux, J. Smith, K. K. Davis, K. Barros, R. Zubatyuk, O. Isayev, and A. E. Roitberg, _Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens_, Journal of Chemical Theory and Computation, 2020, DOI: 10.1021/acs.jctc.0c00121