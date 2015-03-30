package database

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

/**
 * 
 */

object CreateDatabaseFiles {

  // ------------------------------------------------------------------------
  // TrainingInputLine fields:
  // 1)sentid 2)arg1 3)arg2 4)turk voting result for each of 5 relations    
  // ------------------------------------------------------------------------
  case class TrainingInputLine(sentId: String, arg1: String, 
      arg2: String, turkvote5rel: String)
 
  case class TrainingSentence(sentId: String, arg1: String, arg2: String)    
  
  // ------------------------------------------------------------------------
  // LivedInInputLine fields:
  //  1)arg1 2)arg2 3)sentid 4)turk voting result for each of 5 relations    
  // ------------------------------------------------------------------------
  case class LivedInInputLine(arg1: String, arg2: String, sentId: String, 
      description: String, istrue: String, relationId: String, id: Int)
      
  // ------------------------------------------------------------------------
  // SentenceInputLine fields:
  // 1)arg1 2)arg1StartOffset 3)arg1EndOffset 4)arg2 5)arg2StartOffset
  // 6)arg2EndOffset 7)sentId 8)relation 9)confidenceScore
  // 10)sentStartOffset 11)sentEndOffset 12)sentence
  // After the sentence, the features and their wts are listed, as
  // ft1 ftwt1 ft2 ftwt2 etc.  
  // ------------------------------------------------------------------------
  case class SentenceInputLine(arg1: String, arg1StartOffset: String, arg1EndOffset: String,
      arg2: String, arg2StartOffset: String, arg2EndOffset: String, sentId: String, 
      relation: String, confidenceScore: String, sentStartOffset: String, 
      sentEndOffset: String, sentence: String, features: String)
      
  // ------------------------------------------------------------------------
  // LivedInFeaturesInputLine fields:
  // 1)relationId 2)features
  // ------------------------------------------------------------------------
  case class LivedInFeaturesInputLine(relationId: String, feature: String)    
            
  // ----------------------------------------------------------
  // Configuration File - specifies input and output files
  // ----------------------------------------------------------  
  val config = ConfigFactory.load("create-database-files.conf")  
  // Files to read
  val sentencesSourceFilename = config.getString("sentences-source-file")
  val trainingSourceFilename = config.getString("training-source-file")
  // Files to write
  val sentencesFilename = config.getString("sentences-file")
  val livedinFilename = config.getString("lived-in-file")
  val livedinfeaturesFilename = config.getString("lived-in-features-file")

  // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
  val props = new Properties()
  props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref")
  val pipeline = new StanfordCoreNLP(props)
  
  // -----------------------------------------------------------------
  // Main - 
  // -----------------------------------------------------------------
  def main(args: Array[String]) {

    val testSentence = "Barack Obama was born in Hawaii awhile back."
    val document = new Annotation(testSentence)
    pipeline.annotate(document)
    val sentencesTest = document.get(classOf[SentencesAnnotation]).asScala.toList
    
    sentencesTest.foreach(s => {
      val tokens = s.get(classOf[TokensAnnotation]).asScala.toList
      for(token <- tokens){
        val word = token.get(classOf[TextAnnotation])
        val lemma = token.get(classOf[LemmaAnnotation])
        val pos = token.get(classOf[PartOfSpeechAnnotation])
        val dep = token.get(classOf[CollapsedCCProcessedDependenciesAnnotation])
        //val dep = token.get(classOf[BasicDependenciesAnnotation])
        var depText = "" 
        //println("roots size: " + dep.getFirstRoot().toString)
        //for (root <- dep.getRoots.asScala)
        //{  depText = depText + "root" }
        //for (edge <- dep.edgeListSorted.asScala.toList)
        //{  depText = depText + edge.getRelation.toString }
        val ner = token.get(classOf[NamedEntityTagAnnotation])
        println(word + " " + lemma + " " + pos + " " + depText + " " + ner)
      }
    })
   
    System.exit(0)
    
    // -------------------------------------------------------
    // Training Source File
    // -------------------------------------------------------
    val livedinLines = {
     
      val inputFilename = trainingSourceFilename
    
      // Does file exist?
      if (!Files.exists(Paths.get(inputFilename))) {
        System.out.println(s"Training file $inputFilename doesn't exist!  " + s"Exiting...")
        sys.exit(1)
      }

      var count = 0
      
      Source.fromFile(inputFilename).getLines().map(line => {
        val tokens = line.trim.split("\t")
        try{
             val livedinToken = tokens(3).trim.split(",")(2)
             //println("livedinToken: " + livedinToken + livedinToken.contains("lived in neg"))
             //println("livedinToken: " + livedinToken + livedinToken.contains("lived in"))
             
             var livedinBoolean = "unknown"
             if(livedinToken.contains("lived in neg")) livedinBoolean = "f"
             else if(livedinToken.contains("lived in")) livedinBoolean = "t"

             val arg1 = tokens(1)
             val arg2 = tokens(2)
             val sentId = tokens(0)
             val description = arg1 + "-" + arg2
             val relationId = sentId + "-" + description
             count += 1
             //TrainingInputLine(tokens(0), tokens(1), tokens(2), livedinBoolean)
             LivedInInputLine(arg1,arg2,sentId,description,livedinBoolean,relationId,count)
             
        }catch{ 
           // if the line doesn't have 4 tokens separated by tabs
           case e: Exception => LivedInInputLine("badline","1","2","3","4","5",0)
        }
        
      }).toList.filter(l => l.sentId != "badline")
           
    } 
       
    println("livedinLines size: " + livedinLines.size)

    val trainingSentences = for(l <- livedinLines) yield {
       TrainingSentence(l.sentId, l.arg1, l.arg2)
    }

    val uniqueTrainingSentences = trainingSentences.toSet
    //val count = new Array[Int](uniqueTrainingSentences.size)
    //var x = uniqueTrainingSentences.zip(count).toMap
    var numDupesTrainingSentences = collection.mutable.Map[TrainingSentence, Int]().withDefaultValue(0)
    for(ts <- uniqueTrainingSentences){
      numDupesTrainingSentences.update(ts, numDupesTrainingSentences(ts))
    }
    
    println("pairs size: " + numDupesTrainingSentences.size)
    
    println("trainingSentences size: " + trainingSentences.size)    
    println("uniqueTrainingSentences size: " + uniqueTrainingSentences.size)
    
    //var trainingSentencesCount = trainingSentences
    
    val sentenceIds = for( ts <- trainingSentences) yield {
      ts.sentId
    }
    
    println("number of sentences: " + sentenceIds.size)
    println("number of unique sentences: " + sentenceIds.toSet.size)    
    
    // -------------------------------------------------------
    // Sentences Source File
    // -------------------------------------------------------

    val sentenceInputLines = {
     
      val inputFilename = sentencesSourceFilename
    
      // Does file exist?
      if (!Files.exists(Paths.get(inputFilename))) {
        System.out.println(s"Sentences file $inputFilename doesn't exist!  " + s"Exiting...")
        sys.exit(1)
      }
      
      Source.fromFile(inputFilename).getLines().map(line => {
        val tokens = line.trim.split("\t")
        try{
          
          val arg1 = tokens(0)
          val arg2 = tokens(3)
          val sentId = tokens(6)
          
          // -----------------------------------------------------  
          // Write-out features if this is a training sentence
          // -----------------------------------------------------  

          val relationId = TrainingSentence(sentId,arg1,arg2)
          if(numDupesTrainingSentences.contains(relationId)){
            numDupesTrainingSentences.update(relationId, numDupesTrainingSentences(relationId) + 1)
          }
          
          val featuresAndScores = tokens.drop(12)

          val featuresList = for(i <- featuresAndScores.indices if(i % 2 ==0)) yield {
            featuresAndScores(i)
          }
          
          val features = featuresList.mkString("\t")
          
          SentenceInputLine(tokens(0),tokens(1),tokens(2),tokens(3),tokens(4),
            tokens(5),tokens(6),tokens(7),tokens(8),tokens(9),tokens(10),tokens(11), features)
                
        }catch{ 
           // if the line doesn't have 12 tokens separated by tabs
           case e: Exception => SentenceInputLine("badline","1","2","3","4","5","6","7","8","9","10","11","12")
        }
        
      }).toList.filter(l => l.arg1 != "badline" && uniqueTrainingSentences.contains(TrainingSentence(l.sentId,l.arg1,l.arg2)) )  
      
    } 
           
    println("sentenceInputLines size: " + sentenceInputLines.size)
    println("numDupesTrainingSentences: " + numDupesTrainingSentences.values.toSet)    
    println("numDupesTrainingSentences size: " + numDupesTrainingSentences.size)
    
    println("Opening output files for writing")

    // --------------------------------------------------------
    // Check if output files exist already
    // If they do, exit with error message
    // --------------------------------------------------------
    
    // Check if the livedin file exists; if it does, exit with error message
    if (Files.exists(Paths.get(livedinFilename))) {
      System.out.println(s"livedin file $livedinFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the sentences file exists; if it does, exit with error message
    if (Files.exists(Paths.get(sentencesFilename))) {
      System.out.println(s"sentences file $sentencesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }

    // Check if the livedinfeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(livedinfeaturesFilename))) {
      System.out.println(s"livedinfeatures file $livedinfeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // ------------------------------------------------------------
    // livedinFilename - write out
    // ------------------------------------------------------------
    
    // Create PrintWriter's
    
    val livedin = new PrintWriter(livedinFilename)    
    val livedinfeatures = new PrintWriter(livedinfeaturesFilename)
    val sentences = new PrintWriter(sentencesFilename)

    // Write Files
    
    livedinLines.foreach(l => {
      livedin.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.istrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )
     
    var sentencesNoDupes = for(s <- sentenceInputLines) yield {
      
      val numDupes = numDupesTrainingSentences.getOrElse(TrainingSentence(s.sentId,s.arg1,s.arg2),0)

      // write features
      if(numDupes == 1) s
      else SentenceInputLine("dupeLine","1","2","3","4","5","6","7","8","9","10","11","12")          
      
    }
    
    sentencesNoDupes.filter(_.arg1 != "dupeLine")
    
    println("sentencesNoDupes size: " + sentencesNoDupes.size)
    
    //sentencesNoDupes.foreach(s => )
    
    /*sentenceInputLines.foreach(s => {
 
      val numDupes = numDupesTrainingSentences.getOrElse(TrainingSentence(s.sentId,s.arg1,s.arg2),0)

      // write features
      if(numDupes == 1){
        
        val relationId = s.sentId + "-" + s.arg1 + "-" + s.arg2
              
        val features = s.features.split("\t")
        features.foreach(f => {
          livedinfeatures.append(relationId + "\t")     
          livedinfeatures.append(f + "\n")  
          }
        )
        
        sentences.append(s.sentId + "\t" + s.sentence + "\t")
        sentences.append("{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + 
          "\t" + "{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + "\t") 
        sentences.append(s.sentStartOffset + "\t" + s.sentId + "\n")
      
      }

      // write sentences
      //if(numDupes > 0){
      //  sentences.append()
      //}            
      
      }
    )*/
    
    println("Closing output files")
    
    livedin.close()
    livedinfeatures.close()
    sentences.close()
    
  }
  
}
