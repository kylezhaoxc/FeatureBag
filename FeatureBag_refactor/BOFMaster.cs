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
    public class BOFMaster
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
            Console.WriteLine("Generating Layer1 SVM\n");

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
    class L1BOF : IDisposable
    {
        private SVM topLayerSVM; private Matrix<float> topLayerDic;
        //private List<Image<Bgr, byte>> refImagesContainer = new List<Image<Bgr, byte>>();
        int TopLayerNum = 0;
        int j;
        private static int classNum = 50; //number of clusters/classes 
        Matrix<float> labels;
        private static SURFDetector _detector = new SURFDetector(500, false);
        private static BruteForceMatcher<float> _matcher = new BruteForceMatcher<float>(DistanceType.L2);
        private BOWKMeansTrainer bowTrainer = new BOWKMeansTrainer(classNum, new MCvTermCriteria(10, 0.01), 3,
        KMeansInitType.PPCenters);
        private BOWImgDescriptorExtractor<float> bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);
        private Matrix<float> trainingDescriptors;

        public void TrainByClass(List<string> dirs, float[] classes)
        {
            foreach (string dir in dirs)
            {
                foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
                {
                    TopLayerNum++;
                }

            }
            labels = new Matrix<float>(TopLayerNum, 1);
            trainingDescriptors = new Matrix<float>(TopLayerNum, classNum);
            j = 0;
            foreach (string dir in dirs)
            {
                foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
                {
                    Extract(new Image<Bgr, byte>(file.FullName));
                }


            }
            Console.WriteLine("Finished Extracting Features.\n");
            MakeDic();
            int i = 0; ; j = 0;
            foreach (string dir in dirs)
            {
                foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
                {
                    MakeDescriptors(new Image<Bgr, byte>(file.FullName), classes[i]);
                }
                i++;

            }
            Console.WriteLine("Finished Making Descriptors.\n");
        }
        public void Save()
        {
            topLayerSVM = new SVM();
            SVMParams p = new SVMParams();
            p.KernelType = SVM_KERNEL_TYPE.LINEAR;
            p.SVMType = SVM_TYPE.C_SVC;
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, 0.00001);
            topLayerSVM.Train(trainingDescriptors, labels, null, null, p);
            IFormatter formatter = new BinaryFormatter();
            Stream fs = File.OpenWrite(AppDomain.CurrentDomain.BaseDirectory + "\\obj\\dic.xml");
            formatter.Serialize(fs, topLayerDic);

            fs.Dispose();
            topLayerSVM.Save(AppDomain.CurrentDomain.BaseDirectory + "\\obj\\svm.xml");

        }
        private void Extract(Image<Bgr, byte> newRefPic)
        {

            //refImagesContainer.Add(newRefPic);
            bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);


            using (Image<Gray, byte> modelGray = newRefPic.Convert<Gray, Byte>())
            //Detect SURF key points from images 
            using (VectorOfKeyPoint modelKeyPoints = _detector.DetectKeyPointsRaw(modelGray, null))
            //Compute detected SURF key points & extract modelDescriptors 
            using (
            Matrix<float> modelDescriptors = _detector.ComputeDescriptorsRaw(modelGray, null, modelKeyPoints)
            )
            {
                //Add the extracted BoW modelDescriptors into BOW trainer 
                bowTrainer.Add(modelDescriptors);
            }
        }
        private void MakeDic()
        {
            topLayerDic = bowTrainer.Cluster();
            bowDe.SetVocabulary(topLayerDic);
        }
        private void MakeDescriptors(Image<Bgr, byte> newRefPic, float x)
        {
            using (Image<Gray, byte> modelGray = newRefPic.Convert<Gray, Byte>())
            using (VectorOfKeyPoint modelKeyPoints = _detector.DetectKeyPointsRaw(modelGray, null))
            using (Matrix<float> modelBowDescriptor = bowDe.Compute(modelGray, modelKeyPoints))
            {
                //To merge all modelBOWDescriptor into single trainingDescriptors 
                for (int i = 0; i < trainingDescriptors.Cols; i++)
                {
                    trainingDescriptors.Data[j, i] = modelBowDescriptor.Data[0, i];
                }
                labels.Data[j, 0] = x;
                j++;
            }
        }

        //private float LoadAndPredict(Image<Bgr, byte> refPic)
        //{
        //    Image<Gray, byte> testImgGray = refPic.Convert<Gray, byte>();
        //    VectorOfKeyPoint testKeyPoints = _detector.DetectKeyPointsRaw(testImgGray, null);
        //    BOWImgDescriptorExtractor<float> bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);
        //    IFormatter formatter = new BinaryFormatter();
        //    FileStream fs = File.OpenRead(AppDomain.CurrentDomain.BaseDirectory + "obj\\dic.xml");
        //    topLayerDic = (Matrix<float>)formatter.Deserialize(fs);
        //    fs.Dispose();
        //    bowDe.SetVocabulary(topLayerDic);
        //    topLayerSVM = new SVM();
        //    topLayerSVM.Load(AppDomain.CurrentDomain.BaseDirectory + "\\obj\\svm.xml");
        //    Matrix<float> testBowDescriptor = bowDe.Compute(testImgGray, testKeyPoints);
        //    float result = topLayerSVM.Predict(testBowDescriptor);
        //    loaded = true;
        //    return result;
        //}
        private float ImmediatPredict(Image<Bgr, byte> refPic)
        {
            Image<Gray, byte> testImgGray = refPic.Convert<Gray, Byte>();
            VectorOfKeyPoint testKeyPoints = _detector.DetectKeyPointsRaw(testImgGray, null);
            BOWImgDescriptorExtractor<float> bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);
            bowDe.SetVocabulary(topLayerDic);
            Matrix<float> testBowDescriptor = bowDe.Compute(testImgGray, testKeyPoints);
            float result = topLayerSVM.Predict(testBowDescriptor);

            return result;
        }
        public float L1Predict(Image<Bgr, byte> refPic)
        {
            ////if (loaded) 
            return ImmediatPredict(refPic);
            //return LoadAndPredict(refPic);
        }

        public void Dispose()
        {
            //Console.WriteLine("Training Complete, Result Saved.\n");
        }
    }
    class L2BOF : IDisposable
    {
        string directory;
        private SVM SecondLayerSVM; private Matrix<float> SecondLayerDic;
        private List<Image<Bgr, byte>> refImagesContainer = new List<Image<Bgr, byte>>();
        int SecondLayerNum = 0;
        int j;
        private static int classNum = 50; //number of clusters/classes 
        Matrix<float> labels;
        private static SURFDetector _detector = new SURFDetector(500, false);
        private static BruteForceMatcher<float> _matcher = new BruteForceMatcher<float>(DistanceType.L2);
        private BOWKMeansTrainer bowTrainer = new BOWKMeansTrainer(classNum, new MCvTermCriteria(10, 0.01), 3,
        KMeansInitType.PPCenters);
        private BOWImgDescriptorExtractor<float> bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);
        private Matrix<float> trainingDescriptors;

        public void TrainEach(string dir)
        {
            directory = dir;
            foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
            {
                SecondLayerNum++;
            }

            labels = new Matrix<float>(SecondLayerNum, 1);
            trainingDescriptors = new Matrix<float>(SecondLayerNum, classNum);
            foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
            {
                Extract(new Image<Bgr, byte>(file.FullName));
            }

            MakeDic();
            j = 0;

            foreach (FileInfo file in new DirectoryInfo(dir).GetFiles())
            {
                MakeDescriptors(new Image<Bgr, byte>(file.FullName));
            }
            Console.WriteLine();
            SecondLayerSVM = new SVM();
            SVMParams p = new SVMParams();
            p.KernelType = SVM_KERNEL_TYPE.LINEAR;
            p.SVMType = SVM_TYPE.C_SVC;
            p.C = 1;
            p.TermCrit = new MCvTermCriteria(100, 0.00001);
            SecondLayerSVM.Train(trainingDescriptors, labels, null, null, p);
        }
        private void Extract(Image<Bgr, byte> newRefPic)
        {

            refImagesContainer.Add(newRefPic);
            bowDe = new BOWImgDescriptorExtractor<float>(_detector, _matcher);


            using (Image<Gray, byte> modelGray = newRefPic.Convert<Gray, Byte>())
            //Detect SURF key points from images 
            using (VectorOfKeyPoint modelKeyPoints = _detector.DetectKeyPointsRaw(modelGray, null))
            //Compute detected SURF key points & extract modelDescriptors 
            using (
            Matrix<float> modelDescriptors = _detector.ComputeDescriptorsRaw(modelGray, null, modelKeyPoints)
            )
            {
                //Add the extracted BoW modelDescriptors into BOW trainer 
                bowTrainer.Add(modelDescriptors);
            }
        }
        private void MakeDic()
        {
            SecondLayerDic = bowTrainer.Cluster();
            bowDe.SetVocabulary(SecondLayerDic);
        }
        private void MakeDescriptors(Image<Bgr, byte> newRefPic)
        {
            using (Image<Gray, byte> modelGray = newRefPic.Convert<Gray, Byte>())
            using (VectorOfKeyPoint modelKeyPoints = _detector.DetectKeyPointsRaw(modelGray, null))
            using (Matrix<float> modelBowDescriptor = bowDe.Compute(modelGray, modelKeyPoints))
            {
                //To merge all modelBOWDescriptor into single trainingDescriptors 
                for (int i = 0; i < trainingDescriptors.Cols; i++)
                {
                    trainingDescriptors.Data[j, i] = modelBowDescriptor.Data[0, i];
                }
                labels.Data[j, 0] = j + 1;
                j++;
            }
        }
        public float L2Predict(Image<Bgr, byte> refPic)
        {
            Image<Gray, byte> testImgGray = refPic.Convert<Gray, Byte>();
            VectorOfKeyPoint testKeyPoints = _detector.DetectKeyPointsRaw(testImgGray, null);
            Matrix<float> testBowDescriptor = bowDe.Compute(testImgGray, testKeyPoints);
            float result = SecondLayerSVM.Predict(testBowDescriptor);

            return result;
        }
        public void Dispose()
        {
            Console.WriteLine("Finished Generating SVM for class\t" + this.directory.Substring(directory.Length - 1));
        }
    }

}
