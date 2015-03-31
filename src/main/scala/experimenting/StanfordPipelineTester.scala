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
import edu.stanford.nlp.trees.TreeCoreAnnotations._

object StanfordPipelineTester {
  
  def main(args: Array[String]) { 
    
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref")
    val pipeline = new StanfordCoreNLP(props);

    // size 24
    // "root(ROOT-0, suggested-3)",
    // "nsubj(suggested-3, Bernanke-1)",
    // "advmod(suggested-3, also-2)",
    // "mark(broaden-9, that-4)",
    // "det(Administration-8, the-5)",
    // "nn(Administration-8, Federal-6)",
    // "nn(Administration-8, Housing-7)",
    // "nsubj(broaden-9, Administration-8)",
    // "nsubj(let-14, Administration-8)",
    // "ccomp(suggested-3, broaden-9)",
    // "poss(program-12, its-10)",
    // "nn(program-12, insurance-11)",
    // "dobj(broaden-9, program-12)",
    // "ccomp(suggested-3, let-14)",
    // "conj_and(broaden-9, let-14)",
    // "amod(people-16, more-15)",
    // "nsubj(switch-17, people-16)",
    // "ccomp(let-14, switch-17)",
    // "amod(mortgages-20, subprime-19)",
    // "prep_from(switch-17, mortgages-20)",
    // "prep_to(switch-17, cheaper-22)",
    // "advmod(insured-25, federally-24)",
    // "amod(loans-26, insured-25)",
    // "appos(cheaper-22, loans-26)" 
    
    val sentence = "Bernanke also suggested that the Federal Housing Administration broaden its insurance program and let more people switch from subprime mortgages to cheaper, federally insured loans."
    //val sentence = "Barack Obama was born in Honolulu, Hawaii in 1953."
    val document = new Annotation(sentence)
    pipeline.annotate(document)
    val sentences = document.get(classOf[SentencesAnnotation]).asScala.toList

    println("sentences size: " + sentences.size)  
    println("Experimenting with the StanfordPipeline!!")

    val s = sentences(0)
    
    val tokens = s.get(classOf[TokensAnnotation]).asScala.toList
    
    val word = for(token <- tokens) yield { token.get(classOf[TextAnnotation]) }
    val lemma = for(token <- tokens) yield { token.get(classOf[LemmaAnnotation]) }
    val pos = for(token <- tokens) yield { token.get(classOf[PartOfSpeechAnnotation]) }
    val ner = for(token <- tokens) yield { token.get(classOf[NamedEntityTagAnnotation]) }

    val tree = s.get(classOf[TreeAnnotation])
    //println("parse: " + tree.toString)
    
    val dep = s.get(classOf[CollapsedCCProcessedDependenciesAnnotation])
    //val dep = s.get(classOf[BasicDependenciesAnnotation])
    var depText = "" 
    println("dep size: " + dep.getRoots.size)
        
    for (root <- dep.getRoots.asScala){
      //println("root: " + root.toString)
      //println
      println("root(ROOT-0, " + root.word + "-" + root.index() + ")" )
      depText = depText + " root" }
    for (edge <- dep.edgeListSorted.asScala){
      depText = depText + " " + edge.getRelation.toString
      //println("------------------------------------------")
      //println("edge: " + edge.toString) 
      //println("edge dep: " + edge.getDependent().toString())
      //println("edge gov: " + edge.getGovernor().toString())
      //println("edge rel: " + edge.getRelation().toString())
      //println("edge source: " + edge.getSource().toString())
      //println("edge targ: " + edge.getTarget().toString())
      //println("edge wt: " + edge.getWeight().toString())
      //println
      println(edge.getRelation().toString + "(" + edge.getSource().word() + "-" + edge.getSource().index() + 
        ", " + edge.getTarget().word() + "-" + edge.getTarget().index + ")")
      
    }
    
  }

}
