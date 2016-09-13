package com.isoftstone.brain.train;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.BalanceMinibatches;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by agibsonccc on 8/22/16.
 */
public class PreProcess {
    //Images are of format given by allowedExtension -
    protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    protected static final long seed = 12345;

    public static final Random randNumGen = new Random(seed);

     protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int outputNum = 2;


    public static void main(String[] args) throws Exception {
        int rows = 56;
        int cols = 56;

        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory
        //
        //
        File parentDir = new File(args[0]);
        System.out.println(" the data folder is "+args[0]);
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into com.isoftstone.brain.train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(rows,cols,channels,labelMaker);

        //Often there is a need to transforming images to artificially increase the size of the dataset
        //DataVec has built in powerful features from OpenCV
        //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10));
            */

        //You can use the ShowImageTransform to view your images
        //Code below gives you a look before and after, for a side by side comparison
        ImageTransform transform = new MultiImageTransform(randNumGen);

        //Initialize the record reader with the com.isoftstone.brain.train data and the transform chain
        recordReader.initialize(filesInDir,transform);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        int batchSize = 100;
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        final DataNormalization normalization = new ImagePreProcessingScaler();
        normalization.fit(dataIter);
        dataIter.reset();
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder()
                .dataNormalization(normalization).dataSetIterator(dataIter).rootDir(parentDir)
                .miniBatchSize(batchSize).numLabels(2).rootSaveDir(new File(".")).build();
        balanceMinibatches.balance();


    }
    public DataNormalization  init (String dataPath) throws IOException {
        int rows = 56;
        int cols = 56;

        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory
        //
        //
        File parentDir = new File(dataPath);
        System.out.println(" the data folder is "+dataPath);
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into com.isoftstone.brain.train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(rows,cols,channels,labelMaker);

        //Often there is a need to transforming images to artificially increase the size of the dataset
        //DataVec has built in powerful features from OpenCV
        //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10));
            */

        //You can use the ShowImageTransform to view your images
        //Code below gives you a look before and after, for a side by side comparison
        ImageTransform transform = new MultiImageTransform(randNumGen);

        //Initialize the record reader with the com.isoftstone.brain.train data and the transform chain
        recordReader.initialize(filesInDir,transform);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        int batchSize = 100;
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        final DataNormalization normalization = new ImagePreProcessingScaler();
        normalization.fit(dataIter);
        dataIter.reset();
        BalanceMinibatches balanceMinibatches = BalanceMinibatches.builder()
                .dataNormalization(normalization).dataSetIterator(dataIter).rootDir(parentDir)
                .miniBatchSize(batchSize).numLabels(2).rootSaveDir(new File(".")).build();
        balanceMinibatches.balance();
        return normalization;
    }

    public void process (String dataPath) throws IOException {

        int rows = 56;
        int cols = 56;


        File parentDir = new File(dataPath);
        System.out.println(" the data folder is "+dataPath);
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];


        ImageRecordReader recordReader = new ImageRecordReader(rows,cols,channels,labelMaker);
        ImageTransform transform = new MultiImageTransform(randNumGen);

        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData,transform);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);

        ImageRecordReader testrecordReader = new ImageRecordReader(rows,cols,channels,labelMaker);
        ImageTransform testtransform = new MultiImageTransform(randNumGen);
        testrecordReader.initialize(testData,testtransform);
        DataSetIterator testdataIter = new RecordReaderDataSetIterator(testrecordReader, 10, 1, outputNum);




    }

}
