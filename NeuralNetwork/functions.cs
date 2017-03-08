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
        public static double sigmoid(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }
        public static double derevativeOfSigmoid(double value)
        {
            double exp = Math.Pow(Math.E, value);
            return exp / ((1 + exp)*(1+exp));
        }
    }
}
