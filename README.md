# Portfolio
Data science and machine learning portfolio.

## [Climbing Grade Classifier](climbing/README.md)

The [moonboard](https://www.moonboard.com) is a climbing training tool.  Each moonboard is a short identical climbing wall with the same holds, at the same orientation in the same place.  The aim is to predict the climbing grade given only the holds used for a climb.  After scraping the data, I transformed it into a picture representation to take advantage of the spacial distribution of the holds and trained a CNN (convolutional neural network).  I experimented using different loss functions to take advantage of the structure present in the labels (they can be ordered numerically).  Achieved a 71% accuracy with a squared earth mover's distance loss (https://arxiv.org/pdf/1611.05916.pdf) instead of 64% accuracy with the usual cross-entropy loss.

**Result**: achieved 71% accuracy on the test dataset.

**Skills used**: web scraping, data cleaning, visualisation, machine learning, neural networks, convolutional neural networks (CNN)

## [House Price Prediction](house-prices/README.md)

In this project I predict English and Welsh house prices in 2015 from historical data (1995-2014) using only the length of the lease, type of property (flat, terrace etc) and if the house is in London.  The dataset consists of approximately 24 million records in CSV format.  We fit a least squares regression (with elastic net regularisation).

**Result**: I obtain a mean error of £640 on the test (2015) dataset after using the 2014 house prices as a validation dataset to fit the hyperparameters

**Skills used**: data cleaning, machine learning, working with big data, regression

## [SMS Spam Prediction](spam/README.md)

A naive Bayes classifier for detecting spam SMS.

**Result**: I achieved a 97.4% accuracy.

**Skills used**: data processing, naive Bayes classifier

## [Gaussian Process approximator in Bayesian inverse problems](https://github.com/luke321321/inversemcmc)

This code contains an example of using a Gaussian Process approximator in Bayesian inverse problems.  In particular, I use the Gaussian process emulator during a simple MCMC estimating the solution to a PDE and look at two different examples based upon current research.

**Result**: Using a Gaussian process as a marginal approximation is recommended.  This is due to it running quicker than a random approximation with the same theoretical convergence speed.

**Skills used**: inverse problems, approximate Bayesian computation, MCMC, Gaussian processes, inference
