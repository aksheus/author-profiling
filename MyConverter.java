//package weka.api;
//import required classes
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;
import java.io.File;

public class MyConverter {
  
  public static void main(String[] args) throws Exception {
    
    // load ARFF
    ArffLoader loader = new ArffLoader();
    loader.setSource(new File(args[0]));
    Instances data = loader.getDataSet();//get instances object

    // save CSV
    CSVSaver saver = new CSVSaver();
    saver.setInstances(data);//set the dataset we want to convert
    //and save as CSV
    saver.setFile(new File(args[1]));
    saver.writeBatch();
  }
} 

