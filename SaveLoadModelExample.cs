using System;

namespace TFSharpExampleWithoutNuget
{
    public class SaveLoadModelExample
    {
        public static void Run()
        {
            int dataPointCount = 1000;
            int inputSize = 4570;
            int outputSize = 3;
            var random = new Random(0);

            var inputDataset = new double[dataPointCount][];
            var outputDataset = new double[dataPointCount][];

            for (int i = 0; i < dataPointCount; i++)
            {
                inputDataset[i] = new double[inputSize];
                outputDataset[i] = new double[outputSize];

                for (int j = 0; j < inputSize; j++)
                {
                    inputDataset[i][j] = random.NextDouble();
                }
                outputDataset[i][random.Next() % outputSize] = 1;
            }

            TFModel tfModel = new TFModel(inputSize, outputSize, 1, 1, "tfModel.pb", random, false);
            tfModel.fit(inputDataset, outputDataset);
            var predictions = tfModel.predict(inputDataset);
            byte[] persisted = tfModel.PersistModelDefinition();
            tfModel.SaveModelMetadataGraph();


            TFModel model2 = new TFModel(inputSize, outputSize, 1, 1, persisted, random, true);
            var predictions2 = model2.predict(inputDataset);

            int sameFirst = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                for (int j = 0; j < predictions[i].Length; j++)
                {
                    Console.WriteLine(predictions[i][j] + " " + predictions2[i][j]);
                    if (Math.Abs(predictions[i][j] - predictions2[i][j]) >= Math.Pow(10, -15))
                    {
                        Console.WriteLine("err");
                    }
                    else
                    {
                        sameFirst++;
                    }
                }
            }

            Console.WriteLine("FINISH");
            Console.ReadLine();

        }
    }
}