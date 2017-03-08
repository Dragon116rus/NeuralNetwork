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
            NeuralNetwork nn = new NeuralNetwork(2,2,1);

            double[] oo = { 0, 0 };
            double[] oi = { 0, 1 };
            double[] io = { 1, 0 };
            double[] ii = { 1, 1, };
            double[] i = { 1 };
            double[] o = { 0 };
            double[] l = { -1 };
            double learningRate = 0.55;
            double momentumConstant = 0.00;
            Random random = new Random();
            for (int j = 0; j < 3200; j++)
            {
                //learningRate = learningRate / (j + 1);
                //momentumConstant /= (j + 1);
                if (j == 1000)
                    ;
                int rand = random.Next(4);
                if ( rand== 3)
                {
                    nn.train(oi, i, learningRate, momentumConstant);

                }
                if (rand == 2)
                {
                    nn.train(oo, o, learningRate, momentumConstant);

                }
                if (rand == 1)
                {
                    nn.train(io, i, learningRate, momentumConstant);

                }
                if (rand == 0)
                    nn.train(ii, i, learningRate, momentumConstant);
               
                

            }
            foreach (var v in nn.getResult(0,0))
            {
                Console.Write("0 0=");
                Console.WriteLine((int)(v + 0.5));
                Console.WriteLine(v);
            }
            foreach (var v in nn.getResult(1,1))
            {
                Console.Write("1 1=");
                Console.WriteLine((int)(v + 0.5));
                Console.WriteLine(v);
            }
            foreach (var v in nn.getResult(0,1))
            {
                Console.Write("0 1=");
                Console.WriteLine((int)(v + 0.5));
                Console.WriteLine(v);
            }
            foreach (var v in nn.getResult(1, 0))
            {
                Console.Write("1 0=");
                Console.WriteLine((int)(v+0.5));
                Console.WriteLine(v);
            }

        }
    }
}
