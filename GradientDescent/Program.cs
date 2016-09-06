using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;

namespace GradientDescent
{
    class Program
    {
        static double learningRate = double.Parse(ConfigurationManager.AppSettings["LearningRate"]);
        static int totalIteration = int.Parse(ConfigurationManager.AppSettings["TotalIteration"]);

        static CSVProcessor csvProcessor = new CSVProcessor();
        static LinearGradientDescentProcessor linearGDProcessor = new LinearGradientDescentProcessor();
        static List<double[]> dataToProcess = new List<double[]>();
        static double[] initialValue = { 0, 0 }; // {b, gradient}
        static double[] computedValue;

        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Linear Gradient Descent Solver");

                dataToProcess = csvProcessor.GetTrainingData("data.csv", ',');

                Console.WriteLine();
                Console.WriteLine(string.Format("Starting with b = {0} and gradient = {1}", initialValue[0], initialValue[1]));
                Console.WriteLine(string.Format("Initial Relative Error = {0}", linearGDProcessor.ComputeRelativeErrorFromGivenPoints(initialValue[0], initialValue[1], dataToProcess)));

                computedValue = linearGDProcessor.GradientDescentProcess(dataToProcess, initialValue[0], initialValue[1], learningRate, totalIteration);

                Console.WriteLine();
                Console.WriteLine(string.Format("Final value with b = {0} and gradient = {1}", computedValue[0], computedValue[1]));
                Console.WriteLine(string.Format("Final Relative Error = {0}", linearGDProcessor.ComputeRelativeErrorFromGivenPoints(computedValue[0], computedValue[1], dataToProcess)));
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            Console.ReadKey();
        }
    }

    class CSVProcessor
    {
        public List<double[]> GetTrainingData(string fileName, char delimiter)
        {
            List<double[]> data = new List<double[]>();

            using (StreamReader reader = new StreamReader(fileName))
            {
                while (!reader.EndOfStream)
                {
                    var holderValue = reader.ReadLine().Split(delimiter);

                    data.Add(
                        new double[] 
                        {
                            double.Parse(holderValue[0]),
                            double.Parse(holderValue[1])
                        }
                    );
                }
            }

            return data;
        }
    }

    class LinearGradientDescentProcessor
    {
        public double ComputeRelativeErrorFromGivenPoints(double b, double gradient, List<double[]> points)
        {
            double totalError = 0;
            double x, y;

            foreach(double[] point in points)
            {
                x = point[0];
                y = point[1];
                totalError += Math.Pow((y - (gradient * x + b)), 2);
            }

            return totalError / points.Count;
        }

        public double[] GradientDescentProcess(List<double[]> points, double initial_b, double initial_m, double learningRate, int totalIteration)
        {
            double[] computedValue = { initial_b, initial_m };

            for (int iteration = 0; iteration < totalIteration; iteration++)
            {
                computedValue = StepGradient(computedValue[0], computedValue[1], points, learningRate);
            }

            return computedValue;
        }

        private double[] StepGradient(double current_b, double current_m, List<double[]> points, double learningRate)
        {
            double totalPoints = points.Count;
            double stepped_b = 0;
            double stepped_m = 0;
            double new_b;
            double new_m;
            double[] computedValue;

            foreach (double[] point in points)
            {
                double x = point[0];
                double y = point[1];

                stepped_b += -(2 / totalPoints) * (y - ((current_m * x) + current_b));
                stepped_m += -(2 / totalPoints) * x * (y - ((current_m * x) + current_b));
            }

            new_b = current_b - (learningRate * stepped_b);
            new_m = current_m - (learningRate * stepped_m);

            computedValue = new double[] { new_b, new_m };

            return computedValue;
        }
    }
}
