using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NeuralNetwork : ICloneable
    {
        public Neuron[][] network;
        public NeuralNetwork(double learningRate, params int[] countsOfNeuronsOnLayers)
        {
            network = new Neuron[countsOfNeuronsOnLayers.Length][];
            int numberOfOutputLayer = countsOfNeuronsOnLayers.Length - 1;
            int numberOfInputLayer = 0;
            for (int numberOfLayer = 0; numberOfLayer < network.Length; numberOfLayer++)
            {
                int countOfNeurons = countsOfNeuronsOnLayers[numberOfLayer];
                if (numberOfOutputLayer != numberOfLayer)
                {
                    countOfNeurons++;
                }
                network[numberOfLayer] = new Neuron[countOfNeurons];
                for (int numberOfNeuron = 0; numberOfNeuron < countOfNeurons; numberOfNeuron++)
                {
                    if (numberOfLayer == numberOfOutputLayer)
                    {
                        network[numberOfLayer][numberOfNeuron] = new OutputNeuron();
                    }
                    else
                    {
                        int numberOfConstantNeuron = countOfNeurons - 1;
                        if (numberOfNeuron == numberOfConstantNeuron)
                        {
                            network[numberOfLayer][numberOfNeuron] = new ConstantNeuron();
                        }
                        else
                        {
                            if (numberOfLayer != numberOfInputLayer)
                            {
                                network[numberOfLayer][numberOfNeuron] = new HiddenNeuron();
                            }
                            else
                            {
                                network[numberOfLayer][numberOfNeuron] = new InputNeuron();
                            }
                        }
                    }
                }
            }
            initSynapsises(learningRate);
        }
        public double[] activation(params double[] weights)
        {
            cleanWeights();
            initInputWeight(weights);

            int sizeOfOutputLayer = network[network.Length - 1].Length;
            Neuron[] outputLayer = network[network.Length - 1];
            double[] result = new double[sizeOfOutputLayer];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (double)outputLayer[i].weight;
            }
            return result;
        }

        private void initSynapsises(double learningRate)
        {
            int countOfLayers = network.Length;

            for (int numberOfLayer = 0; numberOfLayer < countOfLayers - 1; numberOfLayer++)
            {
                Neuron[] mainLayer = network[numberOfLayer];
                Neuron[] nextLayer = network[numberOfLayer + 1];
                for (int numberOfNeuronOfMainLayer = 0; numberOfNeuronOfMainLayer < mainLayer.Length; numberOfNeuronOfMainLayer++)
                {
                    Neuron neuronOfMainLayer = mainLayer[numberOfNeuronOfMainLayer];

                    int countOfOutSynapsises = nextLayer.Length - countOfConstantNeurons(ref nextLayer);
                    int countOfInSynapsises = mainLayer.Length;

                    neuronOfMainLayer.outSynapsises = new Synapsis[countOfOutSynapsises];


                    for (int numberOfNeuronfOfNextLayer = 0; numberOfNeuronfOfNextLayer < countOfOutSynapsises; numberOfNeuronfOfNextLayer++)
                    {
                        Neuron neuronOfNextLayer = nextLayer[numberOfNeuronfOfNextLayer];
                        if (neuronOfNextLayer.inSynapsises == null)
                        {
                            neuronOfNextLayer.inSynapsises = new Synapsis[countOfInSynapsises];
                        }
                        Synapsis synapsis = new Synapsis(neuronOfMainLayer, neuronOfNextLayer, learningRate);
                        neuronOfMainLayer.outSynapsises[numberOfNeuronfOfNextLayer] = synapsis;
                        neuronOfNextLayer.inSynapsises[numberOfNeuronOfMainLayer] = synapsis;

                    }
                }
            }

        }

        private int countOfConstantNeurons(ref Neuron[] layer)
        {
            int count = 0;
            foreach (var neuron in layer)
            {
                if (neuron is ConstantNeuron)
                {
                    count++;
                }
            }
            return count;
        }

        private void cleanWeights()
        {
            for (int numberOfLayer = 1; numberOfLayer < network.Length; numberOfLayer++)
            {
                for (int numberOfNeuron = 0; numberOfNeuron < network[numberOfLayer].Length; numberOfNeuron++)
                {
                    network[numberOfLayer][numberOfNeuron].weight = null;
                }
            }
        }

        private void initInputWeight(double[] weights)
        {
            if (weights.Length != network[0].Length - 1)
            {
                throw new Exception("Несовпадание длин массивов");
            }
            for (int i = 0; i < weights.Length; i++)
            {
                network[0][i].weight = weights[i];
            }

        }

        public double train(double[] inputData, double[] desiredData, double momentumConstant = 0)
        {
            double[] outputData = activation(inputData);
            double[] errors = getErrorSignal(desiredData, outputData);
            double error = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                error += errors[i] * errors[i];
            }
            error = Math.Sqrt(error / errors.Length);
            updateSynapsisesWeight(errors, momentumConstant);
            return error;
        }

        private double[] getErrorSignal(double[] desiredData, double[] outputData)
        {
            double[] errorSignal = new double[desiredData.Length];
            for (int i = 0; i < errorSignal.Length; i++)
            {
                errorSignal[i] = desiredData[i] - outputData[i];
            }
            return errorSignal;
        }

        private void updateSynapsisesWeight(double[] errors, double momentumConstant)
        {
            getLocalGradientsForOutputLayer(errors);
            getLocalGradientsForHiddenLayer();
            for (int numberOfLayer = network.Length - 1; numberOfLayer > 0; numberOfLayer--)
            {
                Neuron[] layer = network[numberOfLayer];
                for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                {
                    if (layer[numberOfNeuron] is HiddenNeuron)
                    {
                        updateInSynapsisesWeight(layer[numberOfNeuron] as HiddenNeuron, momentumConstant);
                    }
                    else
                    {
                        if (!(layer[numberOfNeuron] is ConstantNeuron))
                        {
                            throw new Exception("Нейрон не того типа");
                        }
                    }
                }
            }
        }

        private void getLocalGradientsForOutputLayer(double[] errors)
        {
            Neuron[] outputLayer = network[network.Length - 1];
            for (int numberOfNeuron = 0; numberOfNeuron < outputLayer.Length; numberOfNeuron++)
            {
                if (outputLayer[numberOfNeuron] is OutputNeuron)
                {
                    OutputNeuron neuron = outputLayer[numberOfNeuron] as OutputNeuron;
                    double localGradient = errors[numberOfNeuron] * neuron.derivativeOfActivationFunction(neuron.inducedLocalField);
                    if (double.IsNaN(localGradient))
                        localGradient = 0;
                    neuron.localGradient = localGradient; 
                }
                else
                {
                    throw new Exception("Нейрон не того типа в выходном слое");
                }
            }
        }

        private void getLocalGradientsForHiddenLayer()
        {
            for (int numberOfLayer = network.Length - 2; numberOfLayer > 0; numberOfLayer--)
            {
                Neuron[] layer = network[numberOfLayer];
                for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                {
                    if (layer[numberOfNeuron] is HiddenNeuron)
                    {
                        HiddenNeuron neuron = layer[numberOfNeuron] as HiddenNeuron;
                        double localGradient= neuron.derivativeOfActivationFunction(neuron.inducedLocalField) *
                            neuron.getDerivivativeOfSqrErrorEnergy();
                        if (double.IsNaN(localGradient))
                            localGradient = 0;
                        neuron.localGradient = localGradient;
                    }
                    else
                    {
                        if (!(layer[numberOfNeuron] is ConstantNeuron))
                        {
                            throw new Exception("Нейрон не того типа в выходном слое");
                        }
                    }
                }
            }
        }

        private void updateInSynapsisesWeight(HiddenNeuron neuron, double momentumConstant)
        {
            int countOfSynapsises = neuron.inSynapsises.Length;
            for (int numberOfSynapsises = 0; numberOfSynapsises < countOfSynapsises; numberOfSynapsises++)
            {
                Synapsis synapsis = neuron.inSynapsises[numberOfSynapsises];
                double delta = momentumConstant * synapsis.prevDeltaWeight +
                    synapsis.learningRate * (double)neuron.localGradient * (double)synapsis.inNeuron.weight;
                synapsis.weight += delta;


                if (Math.Abs(delta + synapsis.prevDeltaWeight) > Math.Abs(delta - synapsis.prevDeltaWeight))
                {
                    synapsis.directory++;
                }
                else
                {
                    synapsis.directory--;
                }
                if (synapsis.directory > 1000)
                {
                    synapsis.directory = 0;
                    synapsis.learningRate *= 1.01;
                }
                if (synapsis.directory < -1000)
                {
                    synapsis.directory = 0;
                    synapsis.learningRate /= 1.01;
                }
                //if (Synapsis.random.Next(1000000) == 153)
                //{
                //    synapsis.weight = Synapsis.random.NextDouble() - 0.5;

                //}
            }
        }
        #region Clone
        public object Clone()
        {
            NeuralNetwork newNetwork = new NeuralNetwork(0.0);
            newNetwork.network = new Neuron[network.Length][];
            createNeurons(ref newNetwork.network);
            copySynapsises(ref newNetwork.network);
            return newNetwork;
        }

        private void copySynapsises(ref Neuron[][] newNeurons)
        {
            int countOfLayers = newNeurons.Length;

            for (int numberOfLayer = 0; numberOfLayer < countOfLayers - 1; numberOfLayer++)
            {
                Neuron[] mainLayer = newNeurons[numberOfLayer];
                Neuron[] nextLayer = newNeurons[numberOfLayer + 1];
                Neuron[] oldMainLayer = network[numberOfLayer];
                for (int numberOfNeuronOfMainLayer = 0; numberOfNeuronOfMainLayer < mainLayer.Length; numberOfNeuronOfMainLayer++)
                {
                    Neuron neuronOfMainLayer = mainLayer[numberOfNeuronOfMainLayer];
                    Neuron neuronOfOldMainLayer = oldMainLayer[numberOfNeuronOfMainLayer];
                    int countOfOutSynapsises = nextLayer.Length - countOfConstantNeurons(ref nextLayer);
                    int countOfInSynapsises = mainLayer.Length;

                    neuronOfMainLayer.outSynapsises = new Synapsis[countOfOutSynapsises];


                    for (int numberOfNeuronfOfNextLayer = 0; numberOfNeuronfOfNextLayer < countOfOutSynapsises; numberOfNeuronfOfNextLayer++)
                    {
                        Neuron neuronOfNextLayer = nextLayer[numberOfNeuronfOfNextLayer];
                        if (neuronOfNextLayer.inSynapsises == null)
                        {
                            neuronOfNextLayer.inSynapsises = new Synapsis[countOfInSynapsises];
                        }
                        double learningRate = neuronOfOldMainLayer.outSynapsises[numberOfNeuronfOfNextLayer].learningRate;
                        double synapsisWeight = neuronOfOldMainLayer.outSynapsises[numberOfNeuronfOfNextLayer].weight;
                        double prevDeltaWeight = neuronOfOldMainLayer.outSynapsises[numberOfNeuronfOfNextLayer].prevDeltaWeight;
                        Synapsis synapsis = new Synapsis(neuronOfMainLayer, neuronOfNextLayer, learningRate, synapsisWeight, prevDeltaWeight);
                        neuronOfMainLayer.outSynapsises[numberOfNeuronfOfNextLayer] = synapsis;
                        neuronOfNextLayer.inSynapsises[numberOfNeuronOfMainLayer] = synapsis;

                    }
                }
            }
        }

        private void createNeurons(ref Neuron[][] newNeurons)
        {
            for (int numberOfLayer = 0; numberOfLayer < network.Length; numberOfLayer++)
            {
                Neuron[] oldNetworkLayer = network[numberOfLayer];

                newNeurons[numberOfLayer] = new Neuron[oldNetworkLayer.Length];
                for (int numberOfNeuron = 0; numberOfNeuron < oldNetworkLayer.Length; numberOfNeuron++)
                {

                    if (oldNetworkLayer[numberOfNeuron] is OutputNeuron)
                    {
                        OutputNeuron oldNeuron = network[numberOfLayer][numberOfNeuron] as OutputNeuron;
                        newNeurons[numberOfLayer][numberOfNeuron] = new OutputNeuron();
                    }
                    else
                    {
                        if (oldNetworkLayer[numberOfNeuron] is HiddenNeuron)
                        {
                            HiddenNeuron oldNeuron = network[numberOfLayer][numberOfNeuron] as HiddenNeuron;
                            newNeurons[numberOfLayer][numberOfNeuron] = new HiddenNeuron();
                        }
                        else
                        {
                            if (oldNetworkLayer[numberOfNeuron] is ConstantNeuron)
                            {
                                newNeurons[numberOfLayer][numberOfNeuron] = new ConstantNeuron();
                            }
                            else
                            {
                                newNeurons[numberOfLayer][numberOfNeuron] = new InputNeuron();
                            }
                        }
                    }
                }

            }
        }
        #endregion Clone
        #region Synapsises Serialize
        public void synapsisesSerialize(string path)
        {
            using (StreamWriter sw = new StreamWriter(path))
            {
                for (int numberOfLayer = 0; numberOfLayer < network.Length; numberOfLayer++)
                {
                    sw.Write(string.Format("{0} ", network[numberOfLayer].Length));
                }
                sw.WriteLine();
                for (int numberOfLayer = 0; numberOfLayer < network.Length-1; numberOfLayer++)
                {
                    Neuron[] layer = network[numberOfLayer];
                    for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                    {
                        Neuron neuron = layer[numberOfNeuron];
                      //  sw.Write("{0}:", neuron.ToString());
                        for (int numberOfSynapsis = 0; numberOfSynapsis < neuron.outSynapsises.Length; numberOfSynapsis++)
                        {
                            Synapsis synapsis = neuron.outSynapsises[numberOfSynapsis];
                            sw.Write(string.Format("{0} {1} {2}:", synapsis.weight,
                                synapsis.prevDeltaWeight,
                                synapsis.learningRate));
                        }
                        sw.Write(";");
                       
                    }
                    sw.WriteLine();
                }
            }
        }
        public void synapsisesDeserialize(string path)
        {
            using (StreamReader sr=new StreamReader(path))
            {
                string line = sr.ReadLine();
                string[] size = line.Split((new char[]{' '}), StringSplitOptions.RemoveEmptyEntries);
                if (size.Length!=network.Length)
                {
                    throw new Exception("Несовпадение размеров сетей");
                }
                for (int numberOfLayer = 0; numberOfLayer < size.Length; numberOfLayer++)
                {
                    int countOfNeuron=0;
                    if (!int.TryParse(size[numberOfLayer],out countOfNeuron) || countOfNeuron!=network[numberOfLayer].Length)
                    {
                        throw new Exception("Несовпадение размеров сетей");
                    }
                }               
                for (int numberOfLayer = 0; numberOfLayer < network.Length - 1; numberOfLayer++)
                {
                    line = sr.ReadLine();
                    string[] serializedLayer = line.Split((new char[] { ';' }), StringSplitOptions.RemoveEmptyEntries);
                    Neuron[] layer = network[numberOfLayer];
                    for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                    {
                        Neuron neuron = layer[numberOfNeuron];
                        string[] serializedNeuron=serializedLayer[numberOfNeuron].Split(new char[] { ':' }, StringSplitOptions.RemoveEmptyEntries);
                        for (int numberOfSynapsis = 0; numberOfSynapsis < neuron.outSynapsises.Length; numberOfSynapsis++)
                        {
                            string[] serializedSynapsis = serializedNeuron[numberOfSynapsis].Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            Synapsis synapsis = neuron.outSynapsises[numberOfSynapsis];
                            double weight, prevDeltaWeight, learningRate;
                            if (double.TryParse(serializedSynapsis[0],out weight) &&
                                double.TryParse(serializedSynapsis[1], out prevDeltaWeight) &&
                                double.TryParse(serializedSynapsis[2], out learningRate))
                            {
                                synapsis.weight = weight;
                                synapsis.prevDeltaWeight = prevDeltaWeight;
                                synapsis.learningRate = learningRate;
                            }
                        }
                        
                    }
                }
            }
        }
        #endregion  Synapsises Serialize
    }
}
