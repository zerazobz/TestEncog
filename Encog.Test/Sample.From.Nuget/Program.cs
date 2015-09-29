using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sample.From.Nuget
{
    class Program
    {
        /// <summary>
        /// Input for XOR Function
        /// </summary>
        public static double[][] XORInput =
        {
            new[] {0.0, 0.0},
            new[] {1.0, 0.0},
            new[] {0.0, 1.0},
            new[] {1.0, 1.0}
        };

        /// <summary>
        /// Ideal Output for the XOR function
        /// </summary>
        public static double[][] XORIdeal =
        {
            new[] {0.0},
            new[] {1.0},
            new[] {1.0},
            new[] {0.0}
        };

        static void Main(string[] args)
        {
            //create a neural network withtout using a factory
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));

            network.Structure.FinalizeStructure();
            network.Reset();

            IMLDataSet trainingSet = new BasicMLDataSet(XORInput, XORIdeal);
            IMLTrain train = new ResilientPropagation(network, trainingSet);

            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine($"Epoch #{epoch} Error: {train.Error}");
                epoch++;
            } while (train.Error > 0.01);
            train.FinishTraining();

            Console.WriteLine("Neural Network Results:");
            foreach (IMLDataPair iPair in trainingSet)
            {
                IMLData output = network.Compute(iPair.Input);
                Console.WriteLine($"{iPair.Input[0]}, {iPair.Input[0]}, actual={output[0]}, ideal={iPair.Ideal[0]}");
            }

            EncogFramework.Instance.Shutdown();

            Console.ReadKey();
        }
    }
}
