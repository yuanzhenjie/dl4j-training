package com.isoftstone.brain.train;

import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Hello world!
 *
 */
public class Training {
    protected static final Logger log = LoggerFactory.getLogger(Training.class);

    //Images are of format given by allowedExtension -
    protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    protected static final long seed = 12345;


    protected static int outputNum = 2;

    public static void main( String[] args ) throws Exception {
        int nChannels = 3;
        int iterations = 1;

        int rows = 56;
        int cols = 56;

//        PreProcess  preProcess = new PreProcess();
//        DataNormalization normalization = preProcess.init(args[0]);
//        DataNormalization normalization = preProcess.init("/Users/yuanzhenjie/Documents/isoftstone-data");
//        System.out.println(" the data folder is "+args[0]);
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.0005)
                .learningRate(1e-3)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder,rows,cols,nChannels);

        MultiLayerNetwork network = new MultiLayerNetwork(builder.build());
        network.setListeners(new ScoreIterationListener(1));
        network.init();


//        String exampleDirectory = "/data/benjamine/projects/dl4j-training/earlystopmodel/";
        String exampleDirectory = "/Users/yuanzhenjie/Documents/workspace/dl4j-training/earlystopmodel/";
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
//                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
                .epochTerminationConditions(new BestScoreEpochTerminationCondition(0.631204948425293)) //Max of 20 minutes
//                .scoreCalculator(new DataSetLossCalculator(testdataIter, true))     //Calculate test set score
                .modelSaver(saver)
                .build();

        EarlyStoppingTrainer trainer =null;
        EarlyStoppingResult result =null;

        DataSet dataSet = new DataSet();
        for(int epoch = 0; epoch < 1; epoch++) {
            File file = new File("/Users/yuanzhenjie/Documents/workspace/dl4j-training/");


            int count =0 ;
            ExistingMiniBatchDataSetIterator existing = new ExistingMiniBatchDataSetIterator(file);
            for (;existing.hasNext();){
                ArrayList  arrayList = new ArrayList();arrayList.add(existing.next());
                esConf.setScoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(arrayList), true));
                trainer = new EarlyStoppingTrainer(esConf,network,new ListDataSetIterator(arrayList));
                result= trainer.fit();
                log.info("*** Completed dataIter.next fit {} ***", count);
                count++;
                System.out.println("Termination reason: " + result.getTerminationReason());
                System.out.println("Termination details: " + result.getTerminationDetails());
                System.out.println("Total epochs: " + result.getTotalEpochs());
                System.out.println("Best epoch number: " + result.getBestModelEpoch());
                System.out.println("Score at best epoch: " + result.getBestModelScore());

                //Print score vs. epoch
                Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
                List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
                Collections.sort(list);
                System.out.println("Score vs. Epoch:");
                for( Integer ii : list){
                    System.out.println(ii + "\t" + scoreVsEpoch.get(epoch));
                }
            }



//
//            File[] files = file.listFiles();
//            int count =0 ;
//            for (File  file1 :files){
//                if(file1.getName().indexOf("dataset")>=0){
////                    dataSet.load(file1);
////                    network.fit(dataSet);
//                    dataSet.load(file1);
//                    ArrayList  arrayList = new ArrayList();arrayList.add(dataSet);
//                    esConf.setScoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(arrayList), true));
//                    trainer = new EarlyStoppingTrainer(esConf,network,dataSet.iterateWithMiniBatches());
//                    result= trainer.fit();
//
//                    log.info("*** Completed dataIter.next fit {} ***", count);
//                    count++;
//                    System.out.println("Termination reason: " + result.getTerminationReason());
//                    System.out.println("Termination details: " + result.getTerminationDetails());
//                    System.out.println("Total epochs: " + result.getTotalEpochs());
//                    System.out.println("Best epoch number: " + result.getBestModelEpoch());
//                    System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//                    //Print score vs. epoch
//                    Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
//                    List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
//                    Collections.sort(list);
//                    System.out.println("Score vs. Epoch:");
//                    for( Integer ii : list){
//                        System.out.println(ii + "\t" + scoreVsEpoch.get(epoch));
//                    }
//                }
//            }

        }

        dataSet.load(new File("dataset-" + 2 + ".bin"));
        Evaluation evaluation = new Evaluation(2);
        evaluation.eval(dataSet.getLabels(),network.output(dataSet.getFeatureMatrix()));
        System.out.println(evaluation.stats());

        ModelSerializer.writeModel(result.getBestModel(),new File("best_model.zip"),true);
        //System.out.println(evaluation.stats());
        ModelSerializer.writeModel(network,new File("model.zip"),true);
//        normalization.save(new File("normalization.bin"),new File("normalization2.bin"));
    }
}
