import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.PrintWriter;
import java.util.Arrays;
public class Layer1 {
    private static Logger log = LoggerFactory.getLogger(pirnapercentageFunct.class);
    public static void main(String[] args) throws  Exception {
        // double lr[] = {0.1};
        double lr[] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
        // double lr[] = {0.1,0.12,0.11,0.1221,0.11119,0.189,0.19,0.144,0.176,0.1444,0.14999,0.15,0.14,0.13,0.134};
        int k ;
        int numLinesToSkip = 0;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("allDataset.txt").getFile()));
        int labelIndex = 12;     //12 values in each row of the allDataset.txt; 12 input features and then class 0/1
        int numClasses = 2;     //2 classes 0/1 
        int batchSize = 1418;    //1418 examples total.
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        KFoldIterator kf = new KFoldIterator(k, allData);
         for (i = 0; i < 5; i++) {
            DataSet trainingData = kf.next();
            DataSet testData = kf.testFold
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();
														/* To normalize our data. We'll use NormalizeStandardize 
														(which gives us mean 0, unit variance): */
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           
            normalizer.transform(trainingData);     
            normalizer.transform(testData);         
            final int numInputs = 12;
            int outputNum = 2;
            int iterations =150;
            long seed = 12345L;
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation(Activation.SIGMOID)
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(lr[j])
                    .updater(Updater.ADAGRAD)
                    .regularization(true).l2(0.001)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(12)
                        .build())
                    .layer(1, new DenseLayer.Builder().nIn(12).nOut(10)
                        .build())
                    .layer(2, new DenseLayer.Builder().nIn(10).nOut(2)
                        .build())
                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(2).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();
               
                MultiLayerNetwork model = new MultiLayerNetwork(conf);      //run the model
                model.init();
                model.setListeners(new ScoreIterationListener(100));

                UIServer uiServer = UIServer.getInstance();
                StatsStorage statsStorage = new InMemoryStatsStorage();             
                int listenerFrequency = 1;
                model.setListeners(new StatsListener(statsStorage, listenerFrequency));
                uiServer.attach(statsStorage);
                model.fit(trainingData);
                
                Evaluation eval = new Evaluation(2);
                INDArray output = model.output(testData.getFeatureMatrix());
                eval.eval(testData.getLabels(), output);
                log.info(eval.stats());

                ROC eval1 = new ROC(2);
                INDArray output = model.output(testData.getFeatureMatrix());
                eval.eval(testData.getLabels(), output);
                eval1.eval(testData.getLabels(), output);
                double a = eval1.calculateAUC();
                log.info(eval1.stats());
		 }
    }
}

