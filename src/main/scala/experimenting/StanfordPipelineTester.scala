package experimenting

import java.io.PrintWriter
import java.nio.file.{Paths, Files}
import collection.JavaConverters._

import scala.collection.mutable._
import scala.io.Source

import com.typesafe.config.ConfigFactory

import java.util.Properties
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.ling.CoreAnnotations._
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations._

object StanfordPipelineTester {
  
  def main(args: Array[String]) { 

    val sentence = "Barack Obama was born in Hawaii in Honolulu, Hawaii in 1953."
    val document = new Annotation(sentence)
    pipeline.annotate(document)
    val sentences = document.get(classOf[SentencesAnnotation]).asScala.toList
  
    println("sentences size: " + sentences.size)
  
  }

}
