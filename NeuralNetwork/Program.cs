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
            NeuralNetwork nn = new NeuralNetwork(2,5,5,5, 2);
            var res=nn.getResult(1, 3);
            foreach(var i in res)
            {
                Console.WriteLine(i);
            }
        }
    }
}
