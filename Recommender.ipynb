{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Recommender \n",
    "\n",
    "This recommender project uses the [movielens data set](https://grouplens.org/datasets/movielens/). It is similar to using the iris or MNIST data set for other algorithms."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# %load /home/ubuntu/projects/initialize.py\n",
    "import findspark\n",
    "findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')\n",
    "import pyspark\n",
    "from pyspark.sql import SparkSession\n",
    "app_name='app'\n",
    "spark=SparkSession.builder.appName(app_name).getOrCreate()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has these parameters:\n",
    "\n",
    "* numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).\n",
    "* rank is the number of latent factors in the model.\n",
    "* iterations is the number of iterations to run.\n",
    "* lambda specifies the regularization parameter in ALS.\n",
    "* implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.\n",
    "* alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "from pyspark.ml.evaluation import RegressionEvaluator\n",
    "from pyspark.ml.recommendation import ALS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "data = spark.read.csv('data/movielens_ratings.csv',inferSchema=True,header=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------+------+------+\n",
      "|movieId|rating|userId|\n",
      "+-------+------+------+\n",
      "|      2|   3.0|     0|\n",
      "|      3|   1.0|     0|\n",
      "|      5|   2.0|     0|\n",
      "|      9|   4.0|     0|\n",
      "|     11|   1.0|     0|\n",
      "|     12|   2.0|     0|\n",
      "|     15|   1.0|     0|\n",
      "|     17|   1.0|     0|\n",
      "|     19|   1.0|     0|\n",
      "|     21|   1.0|     0|\n",
      "|     23|   1.0|     0|\n",
      "|     26|   3.0|     0|\n",
      "|     27|   1.0|     0|\n",
      "|     28|   1.0|     0|\n",
      "|     29|   1.0|     0|\n",
      "|     30|   1.0|     0|\n",
      "|     31|   1.0|     0|\n",
      "|     34|   1.0|     0|\n",
      "|     37|   1.0|     0|\n",
      "|     41|   2.0|     0|\n",
      "+-------+------+------+\n",
      "only showing top 20 rows\n",
      "\n"
     ]
    }
   ],
   "source": [
    "data.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------+------------------+------------------+------------------+\n",
      "|summary|           movieId|            rating|            userId|\n",
      "+-------+------------------+------------------+------------------+\n",
      "|  count|              1501|              1501|              1501|\n",
      "|   mean| 49.40572951365756|1.7741505662891406|14.383744170552964|\n",
      "| stddev|28.937034065088994| 1.187276166124803| 8.591040424293272|\n",
      "|    min|                 0|               1.0|                 0|\n",
      "|    max|                99|               5.0|                29|\n",
      "+-------+------------------+------------------+------------------+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "data.describe().show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can do a split to evaluate how well our model performed, but keep in mind that it is very hard to know conclusively how well a recommender system is truly working for some topics. Especially if subjectivity is involved, for example not everyone that loves star wars is going to love star trek, even though a recommendation system may suggest otherwise."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Smaller dataset so we will use 0.8 / 0.2\n",
    "(training, test) = data.randomSplit([0.8, 0.2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Build the recommendation model using ALS on the training data\n",
    "als = ALS(maxIter=5, regParam=0.01, userCol=\"userId\", itemCol=\"movieId\", ratingCol=\"rating\")\n",
    "model = als.fit(training)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now let's see how the model performed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Evaluate the model by computing the RMSE on the test data\n",
    "predictions = model.transform(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------+------+------+------------+\n",
      "|movieId|rating|userId|  prediction|\n",
      "+-------+------+------+------------+\n",
      "|     31|   4.0|    12|   0.2301091|\n",
      "|     31|   1.0|    19|  0.90208876|\n",
      "|     31|   3.0|     8|   2.1959877|\n",
      "|     85|   1.0|    13|   3.5168304|\n",
      "|     85|   5.0|    16|   3.4186137|\n",
      "|     85|   3.0|    21|   2.3864803|\n",
      "|     65|   1.0|    16|  -1.0209796|\n",
      "|     53|   3.0|    20|   2.0802665|\n",
      "|     78|   1.0|    27|  0.84140104|\n",
      "|     78|   1.0|     2|   -0.833247|\n",
      "|     34|   3.0|    25|   1.3445616|\n",
      "|     34|   1.0|     0| -0.27896202|\n",
      "|     81|   4.0|    11|  -1.9030857|\n",
      "|     81|   3.0|    18|  0.77249664|\n",
      "|     28|   1.0|    27|-0.017428888|\n",
      "|     28|   1.0|     0|   1.5608784|\n",
      "|     76|   1.0|     1|  -2.7133338|\n",
      "|     76|   3.0|     3|   0.5973069|\n",
      "|     26|   1.0|     6| 0.118736714|\n",
      "|     26|   1.0|     3|-0.061883956|\n",
      "+-------+------+------+------------+\n",
      "only showing top 20 rows\n",
      "\n"
     ]
    }
   ],
   "source": [
    "predictions.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Root-mean-square error = 1.8615765938717304\n"
     ]
    }
   ],
   "source": [
    "evaluator = RegressionEvaluator(metricName=\"rmse\", labelCol=\"rating\",predictionCol=\"prediction\")\n",
    "rmse = evaluator.evaluate(predictions)\n",
    "print(\"Root-mean-square error = \" + str(rmse))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The RMSE described our error in terms of the stars rating column."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So now that we have the model, how would we actually supply a recommendation to a user?\n",
    "\n",
    "The same way we did with the test data! For example:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "single_user = test.filter(test['userId']==11).select(['movieId','userId'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------+------+\n",
      "|movieId|userId|\n",
      "+-------+------+\n",
      "|     19|    11|\n",
      "|     22|    11|\n",
      "|     25|    11|\n",
      "|     27|    11|\n",
      "|     30|    11|\n",
      "|     36|    11|\n",
      "|     41|    11|\n",
      "|     45|    11|\n",
      "|     70|    11|\n",
      "|     81|    11|\n",
      "|     90|    11|\n",
      "+-------+------+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# User had 10 ratings in the test data set \n",
    "# Realistically this should be some sort of hold out set!\n",
    "single_user.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "reccomendations = model.transform(single_user)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------+------+-----------+\n",
      "|movieId|userId| prediction|\n",
      "+-------+------+-----------+\n",
      "|     19|    11|  4.6094747|\n",
      "|     25|    11|  3.9128265|\n",
      "|     36|    11|   2.470443|\n",
      "|     45|    11|  1.9146473|\n",
      "|     27|    11|  1.8388556|\n",
      "|     22|    11|   1.420126|\n",
      "|     90|    11| 0.83732325|\n",
      "|     41|    11| 0.70224786|\n",
      "|     70|    11|-0.22689545|\n",
      "|     30|    11| -1.6252961|\n",
      "|     81|    11| -1.9030857|\n",
      "+-------+------+-----------+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "reccomendations.orderBy('prediction',ascending=False).show()"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [Root]",
   "language": "python",
   "name": "Python [Root]"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
