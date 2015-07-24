using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeatureBag_refactor
{
    class Program
    {
        static void Main(string[] args)
        {
            string prefix = AppDomain.CurrentDomain.BaseDirectory + "img\\";
            float[] labels = { 1, 2 };
            List<string> trainingSet = new List<string> { prefix + "c1", prefix + "c2" };
            List<SecondBag> secondbof = new List<SecondBag>();
            foreach (string src in trainingSet)
            {
                SecondBag bag = new SecondBag();
                bag.TrainByClass(src);
                secondbof.Add(bag);
            }

            FirstBag bag1 = new FirstBag();
            bag1.TrainByClass(trainingSet, labels);
            bag1.Save();



            float index_1 = bag1.TopLayerPredict(new Image<Bgr, byte>("D:\\2-test.jpg"));
            float index_1_1 = secondbof[Convert.ToInt32(index_1) - 1].SecondLayerPredict(new Image<Bgr, byte>("D:\\2-test.jpg"));
            float index_2 = bag1.TopLayerPredict(new Image<Bgr, byte>("D:\\1-test.jpg"));
            float index_2_1 = secondbof[Convert.ToInt32(index_2) - 1].SecondLayerPredict(new Image<Bgr, byte>("D:\\1-test.jpg"));
            Console.Write("Index1\n" + index_1 +"-"+index_1_1+ "\nIndex2\n" + index_2 + "-" + index_2_1);
            Console.Read();

        }
    }
}
