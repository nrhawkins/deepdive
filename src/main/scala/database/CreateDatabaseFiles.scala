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
  // in the file:
  // 1)arg1 2)arg2 3)sentid 4)turk voting result for each of 5 relations  
  // save this:
  // 1)arg1 2)arg2 3)sentId 4)description 5)isTrue 6)relationId 7)id    
  // ------------------------------------------------------------------------
  case class TrainingInputLine(arg1: String, arg2: String, 
      sentId: String, description: String, isTrue: String, relationId: String, 
      id: Int)
 
  case class TrainingSentence(sentId: String, arg1: String, arg2: String)    
  
  // ------------------------------------------------------------------------
  // LivedInInputLine fields:
  //  in the file
  //  1)arg1 2)arg2 3)sentid 4)turk voting result for each of 5 relations    
  //  save this:
  // 
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
  
  //val nationalityFilename = config.getString("nationality-file")
  //val borninFilename = config.getString("born-in-file")
  //val livedinFilename = config.getString("lived-in-file")
  //val diedinFilename = config.getString("died-in-file")
  //val traveledtoFilename = config.getString("traveled-to-file")
  
  val featuresFilename = config.getString("features-file")
  //val nationalityfeaturesFilename = config.getString("nationality-features-file")
  //val borninfeaturesFilename = config.getString("born-in-features-file")  
  //val livedinfeaturesFilename = config.getString("lived-in-features-file")
  //val diedinfeaturesFilename = config.getString("died-in-features-file")
  //val traveledtofeaturesFilename = config.getString("traveled-to-features-file")
  
  // Create a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
  val props = new Properties()
  props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref")
  val pipeline = new StanfordCoreNLP(props)
  
  // -----------------------------------------------------------------
  // Main - 
  // -----------------------------------------------------------------
  def main(args: Array[String]) {
    
    // -------------------------------------------------------
    // Training Source File
    // -------------------------------------------------------
    
    val trainingInput = {
    //val (has_nationality, born_in, lived_in, died_in, traveled_to) = {
     
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
             val nationalityToken = tokens(3).trim.split(",")(0)
             val borninToken = tokens(3).trim.split(",")(1)
             val livedinToken = tokens(3).trim.split(",")(2)
             val diedinToken = tokens(3).trim.split(",")(3)
             val traveledtoToken = tokens(3).trim.split(",")(4)
             //println("livedinToken: " + livedinToken + livedinToken.contains("lived in neg"))
             //println("livedinToken: " + livedinToken + livedinToken.contains("lived in"))
             
             var isTrueNationality = "unknown"
             if(nationalityToken.contains("has nationality neg")) isTrueNationality = "f"
             else if(nationalityToken.contains("has nationality")) isTrueNationality = "t"
             
             var isTrueBornIn = "unknown"
             if(borninToken.contains("was born in neg")) isTrueBornIn = "f"
             else if(borninToken.contains("was born in")) isTrueBornIn = "t"   
               
             var isTrueLivedIn = "unknown"
             if(livedinToken.contains("lived in neg")) isTrueLivedIn = "f"
             else if(livedinToken.contains("lived in")) isTrueLivedIn = "t"

             var isTrueDiedIn = "unknown"
             if(diedinToken.contains("died in neg")) isTrueDiedIn = "f"
             else if(diedinToken.contains("died in")) isTrueDiedIn = "t"
               
             var isTrueTraveledTo = "unknown"
             if(traveledtoToken.contains("traveled to neg")) isTrueTraveledTo = "f"
             else if(traveledtoToken.contains("traveled to")) isTrueTraveledTo = "t"
               
             val arg1 = tokens(1)
             val arg2 = tokens(2)
             val sentId = tokens(0)
             val description = arg1 + "-" + arg2
             val relationId = sentId + "-" + description
             count += 1
         
             (TrainingInputLine(arg1,arg2,sentId,description, isTrueNationality, relationId,count),
              TrainingInputLine(arg1,arg2,sentId,description, isTrueBornIn, relationId,count),
              TrainingInputLine(arg1,arg2,sentId,description, isTrueLivedIn, relationId,count),
              TrainingInputLine(arg1,arg2,sentId,description, isTrueDiedIn, relationId,count),
              TrainingInputLine(arg1,arg2,sentId,description, isTrueTraveledTo, relationId,count)
              )
             
        }catch{ 
           // if the line doesn't have 4 tokens separated by tabs
           case e: Exception => 
              (TrainingInputLine("badline","1","2","3","4","5",0),
               TrainingInputLine("badline","1","2","3","4","5",0),
               TrainingInputLine("badline","1","2","3","4","5",0),
               TrainingInputLine("badline","1","2","3","4","5",0),
               TrainingInputLine("badline","1","2","3","4","5",0))
        }
        
        //}).toList.filter((l1,l2,l3,l4,l5) => l1.sentId != "badline")
        
      }).toList
   }        
   
    val has_nationality = for (ti <- trainingInput) yield {ti._1}
    val born_in = for (ti <- trainingInput) yield {ti._2}
    val lived_in = for (ti <- trainingInput) yield {ti._3}
    val died_in = for (ti <- trainingInput) yield {ti._4}
    val traveled_to = for (ti <- trainingInput) yield {ti._5 }

    println("trainingInput size: " + trainingInput.size)
    println("has_nationality size: " + has_nationality.size)    
    println("born_in size: " + born_in.size)
    println("lived_in size: " + lived_in.size)
    println("died_in size: " + died_in.size)
    println("traveled_to size: " + traveled_to.size)
    
    //System.exit(0)
    
    /*val livedinLines = {
     
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
           
    }*/ 
       
    //println("livedinLines size: " + livedinLines.size)

    /*val trainingSentences = for(l <- livedinLines) yield {
       TrainingSentence(l.sentId, l.arg1, l.arg2)
    }

    val uniqueTrainingSentences = trainingSentences.toSet
    var numDupesTrainingSentences = collection.mutable.Map[TrainingSentence, Int]().withDefaultValue(0)
    for(ts <- uniqueTrainingSentences){
      numDupesTrainingSentences.update(ts, numDupesTrainingSentences(ts))
    }
    
    val sentenceIds = for( ts <- trainingSentences) yield {
      ts.sentId
    }*/
    
    /* ----------- UNDO
    
    val trainingSentences = for(l <- lived_in) yield {
       TrainingSentence(l.sentId, l.arg1, l.arg2)
    }
    val uniqueTrainingSentences = trainingSentences.toSet
    var numDupesTrainingSentences = collection.mutable.Map[TrainingSentence, Int]().withDefaultValue(0)
    for(ts <- uniqueTrainingSentences){
      numDupesTrainingSentences.update(ts, numDupesTrainingSentences(ts))
    }    
    val sentenceIds = for( ts <- trainingSentences) yield {
      ts.sentId
    }
    
    
    // 5000 sentences, 4950 unique sentences
    // 5000 trainingSentences, 4997 uniqueTrainingSentences based on triple: (sentid, arg1, arg2), 
    // 4997 entries in map: numDupesTrainingSentences 
    // for writing features file, want to select sentences with only 1 triple (sentid, arg1, arg2),
    // since we can't distinguish between them based on info in these files
    println("number of sentences: " + sentenceIds.size)
    println("number of unique sentences: " + sentenceIds.toSet.size)    
    println("trainingSentences size: " + trainingSentences.size)    
    println("uniqueTrainingSentences size: " + uniqueTrainingSentences.size)
    println("numDupesTrainingSentences size: " + numDupesTrainingSentences.size)    

    ---------------UNDO */

    //System.exit(0)
    
    // -------------------------------------------------------
    // Sentences Source File
    // -------------------------------------------------------

    /* --------------------------- UNDO
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
    
    // --------------------
    // "Has" Relationship
    // --------------------
    
    // Check if the nationality file exists; if it does, exit with error message
    if (Files.exists(Paths.get(nationalityFilename))) {
      System.out.println(s"nationality file $nationalityFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the bornin file exists; if it does, exit with error message
    if (Files.exists(Paths.get(borninFilename))) {
      System.out.println(s"bornin file $borninFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the livedin file exists; if it does, exit with error message
    if (Files.exists(Paths.get(livedinFilename))) {
      System.out.println(s"livedin file $livedinFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }

    // Check if the diedin file exists; if it does, exit with error message
    if (Files.exists(Paths.get(diedinFilename))) {
      System.out.println(s"diedin file $diedinFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the traveledto file exists; if it does, exit with error message
    if (Files.exists(Paths.get(traveledtoFilename))) {
      System.out.println(s"traveledto file $traveledtoFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    ----------------------- UNDO */
    
    // -----------------
    // Sentences
    // -----------------
    
    // Check if the sentences file exists; if it does, exit with error message
    if (Files.exists(Paths.get(sentencesFilename))) {
      System.out.println(s"sentences file $sentencesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // ------------------
    // Features
    // ------------------

    // Check if the features file exists; if it does, exit with error message
    if (Files.exists(Paths.get(featuresFilename))) {
      System.out.println(s"features file $featuresFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    /*// Check if the nationalityfeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(nationalityfeaturesFilename))) {
      System.out.println(s"nationalityfeatures file $nationalityfeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the borninfeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(borninfeaturesFilename))) {
      System.out.println(s"borninfeatures file $borninfeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }
    
    // Check if the livedinfeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(livedinfeaturesFilename))) {
      System.out.println(s"livedinfeatures file $livedinfeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }

    // Check if the diedinfeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(diedinfeaturesFilename))) {
      System.out.println(s"diedinfeatures file $diedinfeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    }

    // Check if the traveledtofeatures file exists; if it does, exit with error message
    if (Files.exists(Paths.get(traveledtofeaturesFilename))) {
      System.out.println(s"traveledtofeatures file $traveledtofeaturesFilename already exists!  " +
        s"\nExiting...")
      sys.exit(1)
    } */
    
    // ------------------------------------------------------------
    // write out files
    // ------------------------------------------------------------
    
    // Create PrintWriter's
    
    // relations
    //val nationality = new PrintWriter(nationalityFilename)    
    //val bornin = new PrintWriter(borninFilename)    
    //val livedin = new PrintWriter(livedinFilename)  
    //val diedin = new PrintWriter(diedinFilename)  
    //val traveledto = new PrintWriter(traveledtoFilename) 
    
    val relationFeatures = new PrintWriter(featuresFilename)
    //val nationalityfeatures = new PrintWriter(nationalityfeaturesFilename)
    //val borninfeatures = new PrintWriter(borninfeaturesFilename)          
    //val livedinfeatures = new PrintWriter(livedinfeaturesFilename)  
    //val diedinfeatures = new PrintWriter(diedinfeaturesFilename)   
    //val traveledtofeatures = new PrintWriter(traveledtofeaturesFilename)
    
    // sentences
    val sentences = new PrintWriter(sentencesFilename)

    // Write Files
    
    /* ---------------------- UNDO
    
    has_nationality.foreach(l => {
      nationality.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.isTrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )

    born_in.foreach(l => {
      bornin.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.isTrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )
    
    lived_in.foreach(l => {
      livedin.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.isTrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )

    died_in.foreach(l => {
      diedin.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.isTrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )

    traveled_to.foreach(l => {
      traveledto.append(l.arg1 + "\t" + l.arg2 + "\t" + l.sentId + "\t" + l.description + 
          "\t" + l.isTrue + "\t" + l.relationId + "\t" + l.id + "\n")      
      }    
    )    
    *    
    *    
    
     
    val sentencesNoDupes = for(s <- sentenceInputLines) yield {
      val numDupes = numDupesTrainingSentences.getOrElse(TrainingSentence(s.sentId,s.arg1,s.arg2),0)
      // write features
      if(numDupes == 1) s
      else SentenceInputLine("dupeLine","1","2","3","4","5","6","7","8","9","10","11","12")                
    }.filter(_.arg1 != "dupeLine")
    
    //sentencesNoDupes = sentencesNoDupes.filter(_.arg1 != "dupeLine")
    
    println("sentencesNoDupes size: " + sentencesNoDupes.size)
    
    sentencesNoDupes.foreach(s => {

      // ------------------
      // write features
      // ------------------
      val relationId = s.sentId + "-" + s.arg1 + "-" + s.arg2
              
        val features = s.features.split("\t")
        features.foreach(f => {
          relationFeatures.append(relationId + "\t")     
          relationFeatures.append(f + "\n")  
          }
        )
        
        // ------------------
        // process sentence
        // ------------------
        val document = new Annotation(s.sentence)
        pipeline.annotate(document)
        val sentence = document.get(classOf[SentencesAnnotation]).asScala.toList(0)
        val tokens = sentence.get(classOf[TokensAnnotation]).asScala.toList
        val word = for(token <- tokens) yield { token.get(classOf[TextAnnotation]) }.mkString(",")
        val lemma = for(token <- tokens) yield { token.get(classOf[LemmaAnnotation]) }.mkString(",")
        val pos = for(token <- tokens) yield { token.get(classOf[PartOfSpeechAnnotation]) }.mkString(",")
        val ner = for(token <- tokens) yield { token.get(classOf[NamedEntityTagAnnotation]) }.mkString(",")      
        val dep = sentence.get(classOf[CollapsedCCProcessedDependenciesAnnotation])
        var depText = ""
        for (root <- dep.getRoots.asScala){
          depText = "root(ROOT-0, " + root.word + "-" + root.index() + ")"
        }
        for (edge <- dep.edgeListSorted.asScala){
          depText = depText + ", " + edge.getRelation().toString + "(" + edge.getSource().word() + "-" + 
          edge.getSource().index() + ", " + edge.getTarget().word() + "-" + edge.getTarget().index + ")"
        }        
        
        // -----------------
        // write sentence
        // -----------------
        sentences.append(s.sentId + "\t" + s.sentence + "\t")
        //sentences.append("{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + 
        //  "\t" + "{\"\",\"\"}" + "\t" + "{\"\",\"\"}" + "\t") 
        sentences.append("\t" + "{" + word + "}" )
        sentences.append("\t" + "{" + lemma + "}" )
        sentences.append("\t" + "{" + pos + "}" )
        sentences.append("\t" + "{" + depText + "}" )
        sentences.append("\t" + "{" + ner + "}" )        
        sentences.append("\t" + s.sentStartOffset + "\t" + s.sentId + "\n" ) 
      
      }
    )
    
    ------------------- UNDO */
    
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
    
    // relations
    //nationality.close()
    //bornin.close()
    //livedin.close()
    //diedin.close()
    //traveledto.close()
    // features
    relationFeatures.close()
    // sentences
    sentences.close()
    
  }
  
}
