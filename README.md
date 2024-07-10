# 🧪 OBIWAN 🧪

This is the code relative to the paper "OBIWAN: An Element-wise Scalable Feed-Forward Neural Network Potential" - [link](https://doi.org/10.1021/acs.jctc.4c00342).

Inspired by the effective ANAKIN method (1), we developed a feed-forward neural network potential whose architecture size doesn't scale with the number of considered species anymore, eventually allowing the user to re-utilise already learnt chemical knowledge without starting the training from scratch everytime.

Together with the training routine and all the other scripts used to produce the results in the article, you can find in [results/models](https://github.com/virtualmartire/OBIWAN/tree/master/results/models) also the checkpoints necessary for the deployment of our model in the two explored scenarios.

## Bibliography

1- C. Devereux, J. Smith, K. K. Davis, K. Barros, R. Zubatyuk, O. Isayev, A. E. Roitberg, _Extending the Applicability of the ANI Deep Learning Molecular Potential to Sulfur and Halogens_, Journal of Chemical Theory and Computation 2020 16(7) 4192–4202, DOI: [10.1021/acs.jctc.0c00121](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00121).