name := "HW_5"

version := "0.1"

scalaVersion := "2.13.7"

val sparkVersion = "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided"
libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())
