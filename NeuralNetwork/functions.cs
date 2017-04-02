using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace NeuralNetwork
{
    class functions
    {
        static double e = Math.E;
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
            //   double exp = Math.Pow(e, 2 * value);
            //  return (exp-1)/(exp+1);
            return Math.Tanh(value);
        }
        public static double derivativeOfTanh(double value)
        {
            //double pExp = Math.Pow(e, value);
            //double mExp = Math.Pow(e, -value);
            //return (pExp-mExp)* (pExp - mExp)/4;
            return Math.Pow(Math.Sinh(value), 2);
        }
        public static double sigmoid(double value)
        {
            return 1 / (1 + Math.Pow(e, -value));
        }
        public static double derevativeOfSigmoid(double value)
        {
            double exp = Math.Pow(e, value);
            return exp / ((1 + exp)*(1+exp));
        }
    }
}
