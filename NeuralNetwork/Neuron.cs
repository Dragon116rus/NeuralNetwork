using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Neuron
    {
        public virtual double? weight { get; set; }
        public Synapsis[] inSynapsises;
        public Synapsis[] outSynapsises;
    }
}
