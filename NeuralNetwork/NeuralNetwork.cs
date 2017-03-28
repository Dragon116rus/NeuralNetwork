using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NeuralNetwork : ICloneable
    {
        public Neuron[][] network;
        public NeuralNetwork(params int[] countsOfNeuronsOnLayers)
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
            initSynapsises();
        }
        public double[] getResult(params double[] weights)
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

        private void initSynapsises()
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
                        Synapsis synapsis = new Synapsis(neuronOfMainLayer, neuronOfNextLayer);
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

        public void train(double[] inputData, double[] desiredData, double learningRate, double momentumConstant = 0)
        {
            double[] outputData = getResult(inputData);
            double[] errors = getErrorSignal(desiredData, outputData);
            updateSynapsisesWeight(errors, learningRate, momentumConstant);
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

        private void updateSynapsisesWeight(double[] errors, double learningRate, double momentumConstant)
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
                        updateInSynapsisesWeight(layer[numberOfNeuron] as HiddenNeuron, learningRate, momentumConstant);
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
                    neuron.localGradient = errors[numberOfNeuron] * neuron.derivativeOfActivationFunction(neuron.inducedLocalField);
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
                        neuron.localGradient = neuron.derivativeOfActivationFunction(neuron.inducedLocalField) *
                            neuron.getDerivivativeOfSqrErrorEnergy();
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

        private void updateInSynapsisesWeight(HiddenNeuron neuron, double learningRate, double momentumConstant)
        {
            int countOfSynapsises = neuron.inSynapsises.Length;
            for (int numberOfSynapsises = 0; numberOfSynapsises < countOfSynapsises; numberOfSynapsises++)
            {
                Synapsis synapsis = neuron.inSynapsises[numberOfSynapsises];
                synapsis.weight += momentumConstant * synapsis.prevDeltaWeight +
                    learningRate * (double)neuron.localGradient* (double)synapsis.inNeuron.weight;
            }
        }

        public object Clone()
        {
            NeuralNetwork newNetwork = new NeuralNetwork();
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
                        double synapsisWeight = network[numberOfLayer][numberOfNeuronOfMainLayer].outSynapsises[numberOfNeuronfOfNextLayer].weight;
                        double prevDeltaWeight = network[numberOfLayer][numberOfNeuronOfMainLayer].outSynapsises[numberOfNeuronfOfNextLayer].prevDeltaWeight;
                        Synapsis synapsis = new Synapsis(neuronOfMainLayer, neuronOfNextLayer, synapsisWeight, prevDeltaWeight);
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
                        newNeurons[numberOfLayer][numberOfNeuron] = new OutputNeuron();
                    }
                    else
                    {
                        if (oldNetworkLayer[numberOfNeuron] is HiddenNeuron)
                        {
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
    }
}
