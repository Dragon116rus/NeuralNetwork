using System;

namespace NeuralNetwork
{
    internal class Synapsis
    {
        private static Random random=new Random();
        public Synapsis(Neuron inNeuron, Neuron outNeuron)
        {
            this.inNeuron = inNeuron;
            this.outNeuron = outNeuron;
            this.weight_ =  (random.NextDouble() - 0.5);
        }
        public Synapsis(Neuron inNeuron, Neuron outNeuron,double weight,double prevDeltaWeight=0)
        {
            this.inNeuron = inNeuron;
            this.outNeuron = outNeuron;
            this.weight_ = weight;
            this.prevDeltaWeight = prevDeltaWeight;
        }
        private double weight_;
        public double weight
        {
            get
            {
                return weight_;
            }
            set
            {
                prevDeltaWeight = weight_-value;
                weight_ = value;
            }
        }
        public double prevDeltaWeight { get;private set; }
        public Neuron inNeuron;
        public Neuron outNeuron;
    }
}