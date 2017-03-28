using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(1,2, 1);

            double learningRate = 0.000000000007;
            double momentumConstant = 0;
            Random random = new Random();
            for (int i = 0; i < 1000000; i++)
            {
                if (i % 100 == 1)
                {
                    getResult(nn);
                }
                double rand = random.Next(15);
                double[] val = new double[1];
                val[0] = rand;
                nn.train(val, val, learningRate, momentumConstant);
            }

        }
        static void getResult(NeuralNetwork nn)
        {
            Random random = new Random();
            double sum = 0;
            for (int i = 0; i < 100; i++)
            {
                double rand = random.Next(15);
                var res = nn.getResult(rand);
                sum += (res[0] - rand) * (res[0] - rand);
            //    Console.WriteLine(string.Format("{0}={1}", rand, res[0]));
            }
            Console.WriteLine(sum/100);
            Console.WriteLine("-------------------------");
        }
    }
}
