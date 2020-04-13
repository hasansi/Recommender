
# Recommender System

With Collaborative filtering we make predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption is that if a user A has the same opinion as a user B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a user chosen randomly.

The image below (from Wikipedia) shows an example of collaborative filtering. At first, people rate different items (like videos, images, games). Then, the system makes predictions about a user's rating for an item not rated yet. The new predictions are built upon the existing ratings of other users with similar ratings with the active user. In the image, the system predicts that the user will not like the video.

<img src=https://upload.wikimedia.org/wikipedia/commons/5/52/Collaborative_filtering.gif />

Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has these parameters:

* numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
* rank is the number of latent factors in the model.
* iterations is the number of iterations to run.
* lambda specifies the regularization parameter in ALS.
* implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
* alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

In this project, I'll use the [movielens data set](https://grouplens.org/datasets/movielens/)
