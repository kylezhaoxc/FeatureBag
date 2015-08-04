using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.ML;
using Emgu.CV.ML.MlEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.Util;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
namespace FeatureBag_refactor
{
    class BOFMaster
    {
        string prefix = AppDomain.CurrentDomain.BaseDirectory + "img\\";
        float[] labels;
        List<string> trainingSet = new List<string>();
        int L1number = 0;
        List<L2BOF> secondbof = new List<L2BOF>();
        L1BOF LayerOneBag;
        public BOFMaster()
        {//calculate class number
            L1number = new DirectoryInfo(prefix).GetDirectories().Length;
               labels = new float[L1number];
            //allocate urls
            for (int i = 0; i < L1number; i++)
            {
                labels[i] = i + 1;
                trainingSet.Add(prefix + "c" + labels[i]);
            }
            Console.WriteLine("Generating Layer2 SVM");
            //train layer2 bofs, add to list
            foreach (string src in trainingSet)
            {
                using (L2BOF bag = new L2BOF())
                {
                    bag.TrainEach(src);
                    secondbof.Add(bag);
                }
                    
            }
            Console.WriteLine();
            Console.WriteLine("Generating Layer1 SVM");

            //train layer1 bof.
            LayerOneBag = new L1BOF();
            LayerOneBag.TrainByClass(trainingSet, labels);
            LayerOneBag.Save();
        }
        public string BOFPredict(Image<Bgr, byte> target)
        {
            float classid = LayerOneBag.L1Predict(target);
            float id = secondbof[Convert.ToInt32(classid) - 1].L2Predict(target);
            string result = classid.ToString() +"-"+ id.ToString();
            return result;
        }

    }
}
