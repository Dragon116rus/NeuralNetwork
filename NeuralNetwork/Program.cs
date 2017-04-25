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

            double learningRate = 0.0015;
            NeuralNetwork nn = new NeuralNetwork(learningRate, 1 ,2, 1);



            nn.train(new double[] { 88 }, new double[] { 88 }, 0);
            nn.synapsisesSerialize("1.txt");

            Console.WriteLine(nn.activation(1)[0]);

            NeuralNetwork nn2 = new NeuralNetwork(learningRate, 1, 25, 25, 1);
            nn2.train(new double[] { 99 }, new double[] { 1 }, 0);
            nn2 = (NeuralNetwork)nn.Clone();

            Console.WriteLine(nn2.activation(1)[0]);

            nn.synapsisesDeserialize("1.txt");

            Console.WriteLine(nn.activation(1)[0]);

            nn = (NeuralNetwork)nn2.Clone();
            Console.WriteLine(nn.activation(1)[0]);



            double momentumConstant = 0.5;
            Random random = new Random();
            for (int i = 0; i < 1; i++)
            {
                if (i % 5000 == 10)
                {
                    getResult(nn);
                }
                double rand = random.Next(15);
                double[] val = new double[1];
                val[0] = rand;
                //  Console.WriteLine(rand);
                //  Console.WriteLine(nn.getResult(rand)[0]);
                nn.train(val, val, momentumConstant);
            }
            nn.synapsisesSerialize("1.txt");
            Console.ReadKey();
        }
        static void getResult(NeuralNetwork nn)
        {
            Random random = new Random();
            double sum = 0;
            for (int i = 0; i < 15; i++)
            {
                double rand = i;//random.Next(15);
                var res = nn.activation(rand);
                sum += (res[0] - rand) * (res[0] - rand);
                Console.WriteLine(string.Format("{0} = {1}", rand, res[0]));
            }
            Console.WriteLine(sum / 100);
            Console.WriteLine("-------------------------");
        }
    }
}
