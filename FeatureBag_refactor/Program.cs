using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime;
using System.Threading;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FeatureBag_refactor
{
    class Program
    {
        static void Main(string[] args)
        {
            BOFMaster searcher = new BOFMaster();

            Stopwatch watch = new Stopwatch();
            watch.Start();
            Console.Write(searcher.BOFPredict(new Image<Bgr, byte>("D:\\2-test.jpg"))+"\t");
            watch.Stop();
            Console.WriteLine( watch.ElapsedMilliseconds + " ms ");
            watch.Restart();
            Console.Write(searcher.BOFPredict(new Image<Bgr, byte>("D:\\1-test.jpg")) + "\t");
            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds + " ms ");
           
            Console.Read();

        }
    }
}
