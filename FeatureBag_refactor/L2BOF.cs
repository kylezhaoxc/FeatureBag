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
    class L2BOF :IDisposable
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
                labels.Data[j, 0] = j+1;
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
            Console.WriteLine("Finished Generating SVM for class\t" + this.directory.Substring(directory.Length-1));
        }
    }
}
