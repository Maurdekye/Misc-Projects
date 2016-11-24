using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuralogical
{
    class Program
    {
        static void Main(string[] args)
        {
            // initialize the network
            Network NumberRecognition = new Network(784, 15, 10);

            // Load in MNIST training and testing data
            Console.WriteLine("Loading training data...");
            Dictionary<byte[,], byte> RawTrainData = Utils.MNIST.Load("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
            Console.WriteLine($"Loaded {RawTrainData.Count} items.");

            Console.WriteLine("Loading testing data...");
            Dictionary<byte[,], byte> RawTestData = Utils.MNIST.Load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
            Console.WriteLine($"Loaded {RawTestData.Count} items.");

            // Convert 2d bytearray data to 1d doublearray data to fit network
            Console.WriteLine("Formatting training data for network...");
            List<Tuple<double[], double[]>> NetworkTrainData = Utils.MNIST.FormatForNetwork(RawTrainData);
            Console.WriteLine("Finished formatting.");

            // Train network with training data
            Console.WriteLine("Training network...");
            NumberRecognition.Train(NetworkTrainData);
            Console.WriteLine("Finished training.");

            // Evaluate network efficacy with testing data
            Console.WriteLine("Testing network...");
            foreach (var image in RawTestData.Keys)
            {
                Tuple<double[], double[]> formatted = Utils.MNIST.FormatForNetwork(image, RawTestData[image]);
                double[] calculated = NumberRecognition.Evaluate(formatted.Item1);
                Console.WriteLine(Utils.MNIST.Show(image));
                Console.WriteLine($"Output from network: {string.Join(" ", calculated)}");
                Console.WriteLine($"Expected output: {string.Join(" ", formatted.Item2)}");
                Console.ReadLine();
            }
        }
    }

    class Network
    {
        int InputLayerCount { get; }
        int HiddenLayerCount { get; }
        int OutputLayerCount { get; }
        private double[,] InputToHiddenWeights;
        private double[,] HiddenToOutputWeights;
        private double[] HiddenLayerBias;
        private double[] OutputLayerBias;

        public Network(int InputLayerCount, int HiddenLayerCount, int OutputLayerCount)
        {
            this.InputLayerCount = InputLayerCount;
            this.HiddenLayerCount = HiddenLayerCount;
            this.OutputLayerCount = OutputLayerCount;

            this.InputToHiddenWeights = Utils.Arrays.InitArray(new double[InputLayerCount, HiddenLayerCount], 1.0);
            this.HiddenToOutputWeights = Utils.Arrays.InitArray(new double[HiddenLayerCount, OutputLayerCount], 1.0);
            this.HiddenLayerBias = Utils.Arrays.InitArray(new double[HiddenLayerCount], 0.5);
            this.OutputLayerBias = Utils.Arrays.InitArray(new double[OutputLayerCount], 0.5);
        }

        public double[] Evaluate(double[] Input)
        {
            if (Input.Length != InputLayerCount)
            {
                throw new IndexOutOfRangeException($"Improper input given; expected a length of {InputLayerCount}, got {Input.Length}");
            }

            // Calculate hidden layer from inputs
            double[] HiddenLayerEval = new double[HiddenLayerCount];
            for (int h=0;h<HiddenLayerCount;h++)
            {
                double weightedSum = 0;
                for (int i=0;i<InputLayerCount;i++)
                {
                    weightedSum += Input[i] * InputToHiddenWeights[i, h];
                }
                HiddenLayerEval[h] = Utils.Maths.Sigmoid(weightedSum - HiddenLayerBias[h]);
            }

            // Calculate output layer from hidden layer calculation
            double[] OutputLayerEval = new double[OutputLayerCount];
            for (int o = 0; o < OutputLayerCount; o++)
            {
                double weightedSum = 0;
                for (int h = 0; h < HiddenLayerCount; h++)
                {
                    weightedSum += HiddenLayerEval[h] * HiddenToOutputWeights[h, o];
                }
                OutputLayerEval[o] = Utils.Maths.Sigmoid(weightedSum - OutputLayerBias[o]);
            }

            return OutputLayerEval;
        }
        
        public void Train(List<Tuple<double[], double[]>> trainingData, double epsilon = 0.01)
        {
            // TODO write training algorithm

        }

        public double EvaluateCurrentCost(List<Tuple<double[], double[]>> trainingData)
        {
            return Cost(trainingData.Select(e => e.Item2).ToList(), trainingData.Select(t => Evaluate(t.Item1)).ToList());
        }

        private static double Cost(List<double[]> ExpectedOutputs, List<double[]> GeneratedOutputs)
        {
            double totalerror = 0;
            for (int i = 0; i < ExpectedOutputs.Count; i++)
            {
                for (int j = 0; j < ExpectedOutputs[i].Count(); j++)
                {
                    double error = ExpectedOutputs[i][j] - GeneratedOutputs[i][j];
                    totalerror += error * error;
                }
            }
            return totalerror / (2 * ExpectedOutputs.Count);
        }
    }
}

namespace Utils
{
    static class Maths
    {
        public static double Sigmoid(double n)
        {
            return 1.0 / (1 + Math.Exp(-n));
        }
    }

    static class Arrays
    {
        public static T[] InitArray<T>(T[] arr, T value)
        {
            for (int x = 0; x < arr.Length; x++)
            {
                arr[x] = value;
            }
            return arr;
        }

        public static T[,] InitArray<T>(T[,] arr, T value)
        {
            for (int x = 0;x < arr.GetLength(0);x++)
            {
                for (int y = 0; y < arr.GetLength(1); y++)
                {
                    arr[x, y] = value;
                }
            }
            return arr;
        }
    }

    public static class MNIST
    {
        public static Dictionary<byte[,], byte> Load(string imageFile, string labelFile)
        {
            byte[] imageFileData = File.ReadAllBytes(imageFile);
            byte[] labelFileData = File.ReadAllBytes(labelFile);
            if (Bytes.BigEndianToInt32(imageFileData, 0) != 2051)
            {
                throw new ArgumentException("Image file provided not correct format");
            }
            if (Bytes.BigEndianToInt32(labelFileData, 0) != 2049)
            {
                throw new ArgumentException("Label file provided not correct format");
            }
            if (Bytes.BigEndianToInt32(labelFileData, 4) != Bytes.BigEndianToInt32(imageFileData, 4))
            {
                throw new ArgumentException("File element lengths do not match");
            }

            Dictionary<byte[,], byte> Images = new Dictionary<byte[,], byte>();
            int numItems = Bytes.BigEndianToInt32(imageFileData, 4);
            int imHeight = Bytes.BigEndianToInt32(imageFileData, 8);
            int imWidth = Bytes.BigEndianToInt32(imageFileData, 12);
            int imOffset = 16;
            int lbOffset = 8;
            
            for (int i = 0; i < numItems; i++)
            {
                byte[,] imData = new byte[imWidth, imHeight];
                for (int x = 0; x < imWidth; x ++)
                {
                    for (int y = 0; y < imHeight; y++)
                    {
                        imData[x, y] = imageFileData[imOffset];
                        imOffset++;
                    }
                }
                Images[imData] = labelFileData[lbOffset];
                lbOffset++;
            }

            return Images;
        }

        public static string Show(byte[,] image, byte threshold = 128, char lowPrint = ' ', char highPrint = '0')
        {
            StringBuilder builder = new StringBuilder();
            for (int x = 0; x < image.GetLength(0); x++)
            {
                for (int y = 0; y < image.GetLength(1); y++)
                {
                    if (image[x, y] > threshold)
                        builder.Append(highPrint);
                    else
                        builder.Append(lowPrint);
                    builder.Append(' ');
                }
                builder.Append('\n');
            }
            return builder.ToString();
        }

        public static Tuple<double[], double[]> FormatForNetwork(byte[,] image, byte label)
        {
            double[] output = Arrays.InitArray(new double[10], 0);
            output[label] = 1;
            
            double[] input = new double[image.GetLength(0) * image.GetLength(1)];
            int index = 0;
            for (int x = 0; x < image.GetLength(0); x++)
            {
                for (int y = 0; y < image.GetLength(1); y++)
                {
                    input[index] = image[x, y] / 255.0;
                    index++;
                }
            }

            return Tuple.Create(input, output);
        }

        public static List<Tuple<double[], double[]>> FormatForNetwork(Dictionary<byte[,], byte> MNISTData)
        {
            List<Tuple<double[], double[]>> dataset = new List<Tuple<double[], double[]>>();
            foreach (var image in MNISTData.Keys)
            {
                dataset.Add(FormatForNetwork(image, MNISTData[image]));
            }
            return dataset;
        }
    }

    public class Bytes
    {
        public static int BigEndianToInt32(byte[] buffer, int offset)
        {
            if (offset < 0)
            {
                throw new ArgumentOutOfRangeException($"Negative offset; {offset}");
            }
            else if (offset >= buffer.Length - 3)
            {
                throw new ArgumentOutOfRangeException($"Offset too large; recieved {offset}, but buffer is only {buffer.Length} long");
            }
            return (buffer[offset] << 24) | (buffer[offset + 1] << 16) | (buffer[offset + 2] << 8) | buffer[offset + 3];
        }
    }
}