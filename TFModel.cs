using System;
using System.IO;
using System.Linq;
using System.Text;
using TensorFlow;

namespace TFSharpExampleWithoutNuget
{
    public class TFModel
    {
        private readonly string STORE_LOAD_TENSOR_NAME = "save/Const:0";
        private readonly string LOAD_TARGET_NAME = "save/restore_all";
        private readonly string STORE_TARGET_NAME = "save/control_dependency";

        private readonly string FILE_PATH = "C:/Users/Snurka/trying_to_save_tf_model/myModel_Csharp";

        private int inputDimension;
        private int outputDimension;
        private int trainingIterations;
        private int batchSize;
        private Random random;
        private TFSession session;
        private double[] trainInputBatch;
        private double[] trainOutputBatch;

        public TFModel(
            int inputDimension,
            int outputDimension,
            int trainingIterations,
            int batchSize,
            string pathToGraphFile,
            Random random,
            bool loadMetadata) : this(inputDimension, outputDimension, trainingIterations, batchSize, File.ReadAllBytes(pathToGraphFile), random, loadMetadata)
        {

        }

        public TFModel(
            int inputDimension,
            int outputDimension,
            int trainingIterations,
            int batchSize,
            byte[] modelAsBytes,
            Random random,
            bool loadMetadata)
        {
            this.inputDimension = inputDimension;
            this.outputDimension = outputDimension;
            this.trainingIterations = trainingIterations;
            this.batchSize = batchSize;
            this.random = random;
            trainInputBatch = new double[batchSize * inputDimension];
            trainOutputBatch = new double[batchSize * outputDimension];

            TFGraph graph = new TFGraph();
            session = new TFSession(graph);

            try
            {
                graph.Import(modelAsBytes);
                if (loadMetadata)
                {
                    session
                        .GetRunner()
                        .AddInput(STORE_LOAD_TENSOR_NAME, TFTensor.CreateString(Encoding.ASCII.GetBytes(FILE_PATH)))
                        .AddTarget(LOAD_TARGET_NAME)
                        .Run();
                }
                else
                {
                    session.GetRunner().AddTarget("init").Run();
                }
            }
            catch (IOException e)
            {
                throw new Exception("Tf model handling crashed. TODO: quite general exception", e);
            }
            Console.WriteLine("Initialized model based on TensorFlow backend.");

            Console.WriteLine(
                "Model with input dimension: [{0}] and output dimension: [{1}]. Batch size of model set to: [{2}]",
                inputDimension, outputDimension, batchSize);
        }

        public void SaveModelMetadataGraph()
        {
            this.session
                .GetRunner()
                .AddInput(STORE_LOAD_TENSOR_NAME, TFTensor.CreateString(Encoding.ASCII.GetBytes(FILE_PATH)))
                .AddTarget(STORE_TARGET_NAME)
                .Run();
        }

        public byte[] PersistModelDefinition()
        {
            TFBuffer buffer = new TFBuffer();
            this.session.Graph.ToGraphDef(buffer);
            return buffer.ToArray();
        }

        public void Dispose()
        {
            Console.WriteLine("Finalizing TF model resources");
            session.CloseSession();
            session.Dispose();
            Console.WriteLine("TF resources closed");
        }

        private void fillbatch(int batchesDone, int[] order, double[][] input, double[][] target)
        {
            // Console.WriteLine("Filling data batch. Already done: [{0}]", batchesDone);
            for (int i = 0; i < batchSize; i++)
            {
                int index = batchesDone * batchSize + i;
                if (index >= order.Length)
                {
                    break; // leaving part of batch from previous iteration.
                }
                Array.Copy(input[order[index]], 0, trainInputBatch, i * inputDimension, inputDimension);
                Array.Copy(target[order[index]], 0, trainOutputBatch, i * outputDimension, outputDimension);
            }
        }

        public void fit(double[][] input, double[][] target)
        {
            if (input.Length != target.Length)
            {
                throw new Exception("Input and target lengths differ");
            }

            Console.WriteLine("Partially fitting TF model on [{0}] inputs.", input.Length);
            int[] order = Enumerable.Range(0, input.Length).ToArray();
            for (int i = 0; i < trainingIterations; i++)
            {
                shuffleArray(order, new Random(random.Next()));
                for (int j = 0; j < (target.Length / batchSize) + 1; j++)
                {
                    fillbatch(j, order, input, target);

                    TFTensor tfInputBatch = TFTensor.FromBuffer(new TFShape(new long[] { batchSize, inputDimension }), trainInputBatch, 0, batchSize * inputDimension);
                    TFTensor tfOutputBatch = TFTensor.FromBuffer(new TFShape(new long[] { batchSize, outputDimension }), trainOutputBatch, 0, batchSize * outputDimension);

                    session
                        .GetRunner()
                        .AddInput("input_node", tfInputBatch)
                        .AddInput("output_node", tfOutputBatch)
                        .AddTarget("train_node")
                        .Run();
                }
            }
        }

        private static void shuffleArray(int[] array, Random rng)
        {
            for (int i = array.Length - 1; i > 0; --i)
            {
                int j = rng.Next(i + 1);
                int temp = array[j];
                array[j] = array[i];
                array[i] = temp;
            }
        }

        public double[] predict(double[] input)
        {
            TFTensor tfInput = TFTensor.FromBuffer(new TFShape(new long[] { 1, input.Length }), input, 0, input.Length);
            TFTensor[] output = session
                .GetRunner()
                .AddInput("input_node", tfInput)
                .Fetch("prediction_node_2")
                .Run();

            TFTensor theFirstAndOnlyFetchedNode = output[0];
            bool jagged = true;
            object native_array_representation = theFirstAndOnlyFetchedNode.GetValue(jagged);  // when jagged = true, returning expected dimensions of return 
            double[][] typed = (double[][])native_array_representation; // typing to expected array with expected dimensions
            return typed[0];  // only one input. so only first output is relevant
        }

        public double[][] predict(double[][] input)
        {
            double[] buffer = new double[input.Length * inputDimension];

            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < inputDimension; j++)
                {
                    buffer[i * inputDimension + j] = input[i][j];
                }
            }

            var inputShape = new TFShape(new long[] { input.Length, inputDimension });

            TFTensor tfInput = TFTensor.FromBuffer(inputShape, buffer, 0, input.Length * inputDimension);

            TFTensor[] output = session
                .GetRunner()
                .AddInput("input_node", tfInput)
                .Fetch("prediction_node_2")
                .Run();

            TFTensor theFirstAndOnlyFetchedNode = output[0];
            bool jagged = true;
            object native_array_representation = theFirstAndOnlyFetchedNode.GetValue(jagged);  // when jagged = true, returning expected dimensions of return 
            double[][] typed = (double[][])native_array_representation; // typing to expected array with expected dimensions
            return typed;
        }

        public int getInputDimension()
        {
            return inputDimension;
        }

        public int getOutputDimension()
        {
            return outputDimension;
        }
    }
}