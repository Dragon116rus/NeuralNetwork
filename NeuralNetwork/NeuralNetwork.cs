using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NeuralNetwork
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
        }
        public double[] getResult(params double[] weights)
        {
            cleanWeights();
            initInputWeight(weights);
            initSynapsises();
            int sizeOfOutputLayer = network[network.Length - 1].Length;
            double[] result = new double[sizeOfOutputLayer];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (double)network[sizeOfOutputLayer - 1][i].weight;
            }
            return result;
        }

        private void initSynapsises()
        {
            int countOfLayers = network.Length;

            for (int numberOfLayer = 0; numberOfLayer < countOfLayers - 1; numberOfLayer++)
            {
                Neuron[] mainLayer = network[numberOfLayer];
                Neuron[] nextLayer = network[numberOfLayer+1];
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

        public void train(double[] inputData, double[] desiredData, double learningDataRate, double momentumConstant = 0)
        {
            double[] outputData = getResult(inputData);
            // double[] errors = getErrorSignal(desiredData, outputData);
            //    updateSynapsisesWeight(errors, learningDataRate, momentumConstant);
        }

        private void updateSynapsisesWeight(double[] errors, double learningDataRate, double momentumConstant)
        {
            getLocalGradientsForOutputLayer(errors);
            getLocalGradientsForHiddenLayer();
            for (int numberOfLayer = 1; numberOfLayer < network.Length; numberOfLayer++)
            {
                Neuron[] layer = network[numberOfLayer];
                for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                {
                    if (layer[numberOfNeuron] is HiddenNeuron)
                    {
                        updateInSynapsisesWeight(layer[numberOfNeuron] as HiddenNeuron, learningDataRate, momentumConstant);
                    }
                    else
                    {
                        throw new Exception("Нейрон не того типа");
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
                    neuron.localGradient = errors[numberOfNeuron] * neuron.derivativeOfActivationFunction(neuron.getInducedLocalField());
                }
                else
                {
                    throw new Exception("Нейрон не того типа в выходном слое");
                }
            }
        }

        private void getLocalGradientsForHiddenLayer()
        {
            for (int numberOfLayer = 1; numberOfLayer < network.Length - 1; numberOfLayer++)
            {
                Neuron[] layer = network[numberOfLayer];
                for (int numberOfNeuron = 0; numberOfNeuron < layer.Length; numberOfNeuron++)
                {
                    if (layer[numberOfNeuron] is HiddenNeuron)
                    {
                        HiddenNeuron neuron = layer[numberOfNeuron] as HiddenNeuron;
                        //  neuron.localGradient=neuron.derivativeOfActivationFunction(neuron.getInducedLocalField())*

                    }
                    else
                    {
                        throw new Exception("Нейрон не того типа в выходном слое");
                    }
                }
            }
        }

        private void updateInSynapsisesWeight(HiddenNeuron neuron, double learningDataRate, double momentumConstant)
        {
            int countOfSynapsises = neuron.inSynapsises.Length;
            for (int numberOfSynapsises = 0; numberOfSynapsises < countOfSynapsises; numberOfSynapsises++)
            {
                Synapsis synapsis = neuron.inSynapsises[numberOfSynapsises];
                synapsis.weight = momentumConstant * synapsis.prevDeltaWeight +
                    learningDataRate * (double)neuron.localGradient * (double)neuron.weight;
            }
        }
    }
}
