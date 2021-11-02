import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite, sum}
import breeze.numerics.{pow, sqrt}
import java.io.{File, FileWriter}


class linReg {
  def fit(X: DenseMatrix[Double], y: DenseVector[Double],
          nIterations: Int, learnRate: Double): (DenseVector[Double], Double) = {
    var (weights, bias) = (DenseVector.zeros[Double](X.cols), .0)
    for (_ <- 0 to nIterations) {
      val yHat = (X * weights) + bias
      weights :-= learnRate * 2 * (X.t * (yHat - y))
      weights = weights.map(el => el / X.rows)
      bias -= learnRate * 2 * sum(yHat - y) / X.rows
    }
    (weights, bias)
  }

  def predict(X: DenseMatrix[Double], weights: DenseVector[Double],
              bias: Double): DenseVector[Double] = {
    (X * weights) + bias
  }

  def RMSE(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    val error = sum((yTrue - yPred).map(el => pow(el, 2))) / yTrue.length
    sqrt(error)
  }
}


object Main {
  def main(args: Array[String]): Unit = {
    val (nIterations, learningRate) = (10000, 0.001)
    val model =  new linReg
    val log = new FileWriter(new File("result/log.txt" ))
    val xTrain = csvread(new java.io.File("data/X_train.csv"))
    val xTest = csvread(new java.io.File("data/X_test.csv"))
    val yTrain = csvread(new java.io.File("data/y_train.csv")).toDenseVector
    val yTest = csvread(new java.io.File("data/y_test.csv")).toDenseVector
    val (weights, bias) = model.fit(xTrain, yTrain, nIterations, learningRate)
    val yPredict = model.predict(xTest, weights, bias)
    csvwrite(new File("result/Predictions.csv"), separator = ',', mat = yPredict.toDenseMatrix)
    log.write(s"RMSE: ${model.RMSE(yTest, yPredict)}.\n")
    log.write(s"Predictions: result/Predictions.csv")
    log.close()
  }
}
