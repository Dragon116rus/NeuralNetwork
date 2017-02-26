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
            NeuralNetwork nn = new NeuralNetwork(1,1,1);

            double[] oo = { 0 };
            for (int i = 0; i < 5; i++)
            {
                
                var res = nn.getResult(0);
                foreach (var b in res)
                {
                    Console.WriteLine(b);
                }
                nn.train(oo, oo, 0.5);
            }
        }
    }
}
