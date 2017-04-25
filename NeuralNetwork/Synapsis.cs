using System;

namespace NeuralNetwork
{
    internal class Synapsis
    {
        public static Random random = new Random();
        public Synapsis(Neuron inNeuron, Neuron outNeuron, double learningRate)
        {
            this.learningRate = learningRate;
            this.inNeuron = inNeuron;
            this.outNeuron = outNeuron;
            this.weight_ = (random.NextDouble() - 0.5);
        }
        public Synapsis(Neuron inNeuron, Neuron outNeuron, double learningRate, double weight, double prevDeltaWeight = 0)
        {
            this.inNeuron = inNeuron;
            this.outNeuron = outNeuron;
            this.weight_ = weight;
            this.prevDeltaWeight = prevDeltaWeight;
        }
        private double weight_;
        public double learningRate;
        public double weight
        {
            get
            {
                return weight_;
            }
            set
            {
                prevDeltaWeight = weight_ - value;
                weight_ = value;
            }
        }
        public double prevDeltaWeight { get; set; }
        public int directory = 0;
        public Neuron inNeuron;
        public Neuron outNeuron;
    }
}