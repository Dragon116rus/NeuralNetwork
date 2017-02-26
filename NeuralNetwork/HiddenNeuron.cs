﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class HiddenNeuron : Neuron
    {
        public HiddenNeuron()
        {
            activationFunction = functions.linearWithoutKoef;
            derivativeOfActivationFunction = functions.derivativeOfLinearWithoutKoef;
        }
        public delegate double ActivationFunction(double value);
        public ActivationFunction activationFunction;
        public ActivationFunction derivativeOfActivationFunction;
        private double? weight_;
        public double? localGradient { get; set; }
        public double getInducedLocalField()
        {
            double sum = 0;
            if (inSynapsises != null)
            {
                foreach (var synapsis in inSynapsises)
                {
                    sum += (double)(synapsis.weight * synapsis.inNeuron.weight);
                }
            }
            return sum;
        }
        public double getDerivivativeOfSqrErrorEnergy()
        {
            double sum = 0;
            foreach (var synapsis in outSynapsises)
            {
                if (synapsis.outNeuron is HiddenNeuron)
                {
                    sum += synapsis.weight * (double)(synapsis.outNeuron as HiddenNeuron).localGradient;
                }
                else
                {
                    throw new Exception("Не тот тип нейрона");
                }
            }
            return sum;
        }
        public override double? weight
        {
            get
            {
                if (weight_ != null)
                    return weight_;
                else
                    return proccessWeight();
            }
            set
            {
                if (value == null)
                {
                    weight_ = null;
                }
                else
                    throw new Exception("Данное присвоение невозможно");
            }
        }
        private double proccessWeight()
        {
            double sum=0;
            foreach(var synapsis in inSynapsises)
            {
                sum += (double)(synapsis.weight * synapsis.inNeuron.weight);
            }
            return activationFunction(sum);
        }
    }
}
