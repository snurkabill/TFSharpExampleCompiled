using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFSharpExampleWithoutNuget
{
    class Benchmark
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
            
            var sw = new Stopwatch();

            for (int i = 0; i < 100; i++)
            {
                sw.Reset();
                sw.Start();
                double[][] output = tfModel.predict(inputDataset);
                sw.Stop();
                Console.WriteLine("Predicting all at once ms: " + sw.ElapsedMilliseconds);
                sw.Reset();
                sw.Start();
                foreach (var entry in inputDataset)
                {
                    double[] prediction = tfModel.predict(entry);
                }
                sw.Stop();
                Console.WriteLine("Predicting one by one ms: " + sw.ElapsedMilliseconds);
                sw.Reset();
                sw.Start();
                tfModel.fit(inputDataset, outputDataset);
                sw.Stop();
                Console.WriteLine("Training ... well this is based on number of iterations in model ms: " + sw.ElapsedMilliseconds);
                sw.Reset();
            }
        }

    }
}
