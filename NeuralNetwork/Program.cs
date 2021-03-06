﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static double[][] samples;
        static double[][] answers;
        static NeuralNetwork nn;
        static string serializePath = "2.txt";
        static void Main(string[] args)
        {

            double learningRate = 0.015;
            nn = new NeuralNetwork(learningRate, 41, 41, 20, 2);


          //  Console.WriteLine("deserialization");
        //   nn.synapsisesDeserialize("2.txt");

            Console.WriteLine("loanding samples");
            loadSamples(5001, 1001);

            Console.WriteLine("training");
            training(10, 1);

            Console.WriteLine("testing");
            testing();
            // show();
            Console.ReadKey();
        }
        private static void show()
        {
            using (StreamReader sr = new StreamReader("C:\\Users\\Dragon116rus\\Downloads\\kddcup.data_10_percent\\kddcup.data.corrected"))
            {
                string line;
                for (int i = 0; i < 60000; i++)
                {
                    sr.ReadLine();
                }
                for (int i = 0; i < 10000; i++)
                {
                    line = sr.ReadLine();
                    Console.WriteLine(line);
                }
                //    while ((line = sr.ReadLine()) != null)
                //       Console.WriteLine(line);
            }
        }
        private static void testing()
        {
            int ddosGood = 0;
            int ddosInvalid = 0;
            int all = 0;
            int normalGood = 0;
            int normalInvalid = 0;
            int unknown = 0;
            double[] answer;
            double[] sample;
            bool isDdos = false;
            using (StreamReader sr = new StreamReader("C:\\Users\\Dragon116rus\\Downloads\\kddcup.data_10_percent\\corrected\\corrected"))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    string[] features = line.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    if (features.Length < 1)
                    {
                        break;
                    }

                    string answerText = features[features.Length - 1];
                    if (answerText == "back." || answerText == "neptune." || answerText == "pod." ||
                        answerText == "smurf." || answerText == "teardrop." || answerText == "land.")
                    {
                        answer = new double[2] { 0, 1 };
                        isDdos = true;
                    }
                    else
                    {
                        isDdos = false;
                        answer = new double[2] { 1, 0 };
                    }
                    sample = new double[41];
                    double value = double.Parse(features[0]);
                    if (value > 1.01)
                    {
                        value = Math.Log(value, 2) / 65531;
                    }
                    sample[0] = value;
                    if (features[1] == "tcp")
                    {
                        sample[1] = 1;
                    }
                    if (features[1] == "udp")
                    {
                        sample[2] = 1;
                    }
                    if (features[1] == "icmp")
                    {
                        sample[3] = 1;
                    }

                    for (int j = 4; j < features.Length - 1; j++)
                    {
                        value = double.Parse(features[j], CultureInfo.InvariantCulture);
                        if (value > 1.01)
                        {
                            value = Math.Log(value, 2) / 65531;
                        }
                        sample[j] = value;
                    }
                    double[] result = nn.activation(sample);
                    //    if (all % 1000 == 0)
                    //       Console.WriteLine("{0}:{1} answer={2}:{3}\n{4}", result[0], result[1], answer[0], answer[1], features[features.Length - 1]);
                    if (Math.Round(result[0]) == 0 && Math.Round(result[1]) == 1)
                    {
                        if (isDdos)
                        {
                            ddosGood++;
                        }
                        else
                        {
                            ddosInvalid++;
                        }
                    }
                    else
                    {
                        if (Math.Round(result[0]) == 1 && Math.Round(result[1]) == 0)
                        {
                            if (!isDdos)
                            {
                                normalGood++;
                            }
                            else
                            {
                                normalInvalid++;
                            }
                        }
                        else
                        {
                            unknown++;
                        }
                    }

                    all++;
                    if (all % 1000 == 0)
                    {
                        Console.WriteLine("all:{0}", all);
                        Console.WriteLine("normal detected:{0} normal invalid:{1}", normalGood, normalInvalid);
                        Console.WriteLine("ddos detected:{0} ddos invalid:{1}", ddosGood, ddosInvalid);
                        Console.WriteLine("unknown:{0}", unknown);
                    }
                }
            }
        }

        private static void training(int epochs, int frequencyOfSerialization = 10)
        {

            for (int i = 0; i < epochs; i++)
            {
                double error = 0;
                int length = samples.Length;
                int[] shuffeledArray = new int[length];
                for (int j = 0; j < length; j++)
                {
                    shuffeledArray[j] = j;
                }
                Random rand = new Random();
                shuffeledArray = shuffeledArray.OrderBy(item => rand.Next()).ToArray();
                for (int j = 0; j < length; j++)
                {
                    int rnd = shuffeledArray[j];
                    error += nn.train(samples[rnd], answers[rnd]);
                    if (j % 1000 == 0)
                    {
                        Console.WriteLine("training: epoch: {0}, completed: {1} / {2}", i, j, length);
                    }
                }
                Console.WriteLine("{0} error={1}", i, error / length);
                if (i % frequencyOfSerialization == 0)
                {
                    Console.WriteLine("serialization");
                    nn.synapsisesSerialize(serializePath);
                }
            }
        }

        private static void loadSamples(int normalSamples, int countOfEachDdosAttacks)
        {
            samples = new double[normalSamples + countOfEachDdosAttacks * 6][];
            answers = new double[normalSamples + countOfEachDdosAttacks * 6][];
            SortedDictionary<string, int> ddosSamplesCount = new SortedDictionary<string, int>();
            using (StreamReader sr = new StreamReader("C:\\Users\\Dragon116rus\\Downloads\\kddcup.data_10_percent\\kddcup.data.corrected"))
            {
                int i = 0;
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    string[] features = line.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    if (features.Length < 1)
                    {
                        break;
                    }

                    string answer = features[features.Length - 1];
                    if (answer == "back." || answer == "neptune." || answer == "pod." ||
                        answer == "smurf." || answer == "teardrop." || answer == "land.")
                    {
                        if (ddosSamplesCount.ContainsKey(answer))
                        {
                            if (ddosSamplesCount[answer] >= countOfEachDdosAttacks)
                            {
                                continue;
                            }
                          
                            ddosSamplesCount[answer]++;
                        }
                        else
                        {
                            ddosSamplesCount.Add(answer, 1);
                        }
                        answers[i] = new double[2] { 0, 1 };
                    }
                    else
                    {
                        if (normalSamples <= 0)
                        {
                            continue;
                        }
                        answers[i] = new double[2] { 1, 0 };
                        normalSamples--;
                    }
                    samples[i] = new double[41];
                    double value = double.Parse(features[0]);
                    if (value > 1.01)
                    {
                        value = Math.Log(value, 2) / 65531;
                    }
                    samples[i][0] = value;
                    if (features[1] == "tcp")
                    {
                        samples[i][1] = 1;
                    }
                    if (features[1] == "udp")
                    {
                        samples[i][2] = 1;
                    }
                    if (features[1] == "icmp")
                    {
                        samples[i][3] = 1;
                    }

                    for (int j = 4; j < features.Length - 1; j++)
                    {
                        value = double.Parse(features[j], CultureInfo.InvariantCulture);
                        if (value > 1.01)
                        {
                            value = Math.Log(value, 2) / 65531;
                        }
                        samples[i][j] = value;
                    }
                    if (countOfEachDdosAttacks == 0 && normalSamples == 0)
                    {
                        break;
                    }
                    i++;
                }
                Array.Resize(ref samples, i-1);
                Array.Resize(ref answers, i-1);
            }
        }


    }
}
