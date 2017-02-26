using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class ConstantNeuron : Neuron
    {
        public override double? weight
        {
            get
            {
                return 1;
            }
            set {; }
        }
    }
}
