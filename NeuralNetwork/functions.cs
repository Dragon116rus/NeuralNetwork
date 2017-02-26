using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class functions
    {
        public static double linearWithoutKoef(double value)
        {
            return value;
        }
        public static double derivativeOfLinearWithoutKoef(double value)
        {
            return 1;
        }
        public static double tanh(double value)
        {
            return Math.Tanh(value);
        }
        public static double derivativeOfTanh(double value)
        {
            return Math.Pow(Math.Sinh(value),2);
        }
    }
}
