using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class OutputNeuron:HiddenNeuron
    {
        public OutputNeuron() {
            activationFunction = functions.linearWithoutKoef;
            derivativeOfActivationFunction = functions.derivativeOfLinearWithoutKoef;
        }
    }
}
