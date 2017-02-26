namespace NeuralNetwork
{
    internal class Synapsis
    {
        public Synapsis(Neuron inNeuron, Neuron outNeuron, double weight = 1)
        {
            this.inNeuron = inNeuron;
            this.outNeuron = outNeuron;
            this.weight_ = weight;
            id1 = id;
            id++;
        }
        private static int id=0;
        private int id1;
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