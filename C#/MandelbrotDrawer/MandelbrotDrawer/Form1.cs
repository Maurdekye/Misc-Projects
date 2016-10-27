using System;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;

namespace MandelbrotDrawer
{
    public partial class Form1 : Form
    {
        public readonly uint Threads = 24;
        public readonly double ZoomScaleFactor = 1.125;
        public readonly Imaginary DefaultLowerBounds = new Imaginary(-2, -1.2);
        public readonly Imaginary DefaultUpperBounds = new Imaginary(2, 1.2);
        public readonly int DefaultCalculationIterations = 256;
        public readonly int DigitsAfterDecimal = 4;

        public Imaginary LowerBounds;
        public Imaginary UpperBounds;
        public int CalculationIterations;
        public Point MandelbrotPanelSize;
        public Imaginary JuliaSubPosition;
        public Thread GlobalRenderThread;

        // Initializing Constructor

        public Form1()
        {
            LowerBounds = DefaultLowerBounds;
            UpperBounds = DefaultUpperBounds;
            CalculationIterations = DefaultCalculationIterations;
            MandelbrotPanelSize = new Point(0, 0);
            InitializeComponent();
        }

        /// Independent Methods

        private Imaginary GetMousePositionOnGrid(Point p)
        {
            double newX = Util.Transform.FromPixel(LowerBounds.Real, UpperBounds.Real, p.X, MandelbrotPanelSize.X);
            double newY = Util.Transform.FromPixel(LowerBounds.Imag, UpperBounds.Imag, p.Y, MandelbrotPanelSize.Y);
            return new Imaginary(newX, newY);
        }

        private void WriteConsoleText(string text)
        {
            Invoke(new Action(() => ConsoleBox.AppendText(text + " \n")));
        }

        /// Event Handlers
        /// 

        // 2nd Tab "Calculate" button

        private void button1_Click(object sender, EventArgs e)
        {
            double realpart;
            double imagpart;
            if (textBox1.Text == "")
                textBox1.Text = "0";
            if (textBox2.Text == "")
                textBox2.Text = "0";
            if (!Double.TryParse(textBox1.Text, out realpart) 
                || !Double.TryParse(textBox2.Text, out imagpart))
                label1.Text = "Both boxes must contain numbers.";
            else
            {
                Imaginary result = new Imaginary(realpart, imagpart);
                label1.Text = result.ToString();
            }
        }

        // Display Panel for fractal render

        private void DisplayBox_Paint(object sender, PaintEventArgs e)
        {
            MandelbrotPanelSize = new Point(DisplayBox.Width, DisplayBox.Height);
            if (MandelbrotPanelSize.X <= 0 || MandelbrotPanelSize.Y <= 0)
                return;
            MandelRendererController render = new MandelRendererController(
                LowerBounds, UpperBounds, MandelbrotPanelSize, JuliaSubPosition, CalculationIterations, Threads);
            render.Activate();
            if (GlobalRenderThread != null && GlobalRenderThread.IsAlive)
                GlobalRenderThread.Abort();
            GlobalRenderThread = new Thread(delegate() {
                render.Compile(DisplayBox);
                WriteConsoleText((render.Timer.ElapsedMilliseconds / 1000.0).ToString() + " Seconds");
            });
            GlobalRenderThread.Start();
        }

        private void DisplayBox_MouseClick(object sender, MouseEventArgs e)
        {
            Imaginary mouseLoc = GetMousePositionOnGrid(e.Location);
            if (e.Button == MouseButtons.Left)
            {
                if (JuliaSubPosition == null)
                {
                    WriteConsoleText(mouseLoc.ToString(DigitsAfterDecimal));
                    JuliaSubPosition = mouseLoc;
                }
                else
                    JuliaSubPosition = null;
                Refresh();
            }
            else if (e.Button == MouseButtons.Right)
            {
                WriteConsoleText(mouseLoc.ToString(DigitsAfterDecimal));
            }
            else if (e.Button == MouseButtons.Middle)
            {
                LowerBounds = DefaultLowerBounds;
                UpperBounds = DefaultUpperBounds;
                CalculationIterations = DefaultCalculationIterations;
                Refresh();
            }
        }

        private void DisplayBox_MouseWheel(object sender, MouseEventArgs e)
        {
            int d = -8*Math.Sign(e.Delta);
            Imaginary mouseLoc = GetMousePositionOnGrid(e.Location);
            Imaginary sizeRange = new Imaginary(UpperBounds.Real - LowerBounds.Real, UpperBounds.Imag - LowerBounds.Imag);
            double horizontalPercentage = (mouseLoc.Real - LowerBounds.Real) / sizeRange.Real;
            double verticalPercentage = (mouseLoc.Imag - LowerBounds.Imag) / sizeRange.Imag;
            double multiplier = Math.Pow(ZoomScaleFactor, d);
            sizeRange.Real *= multiplier;
            sizeRange.Imag *= multiplier;
            LowerBounds = new Imaginary(mouseLoc.Real - sizeRange.Real * horizontalPercentage, 
                mouseLoc.Imag - sizeRange.Imag * verticalPercentage);
            UpperBounds = new Imaginary(mouseLoc.Real + sizeRange.Real * (1 - horizontalPercentage), 
                mouseLoc.Imag + sizeRange.Imag * (1 - verticalPercentage));
            CalculationIterations = Math.Max(CalculationIterations - d*2, 1);
            Refresh();
        }

        private void ConsoleBox_KeyDown(object sender, KeyEventArgs e)
        {

        }

        private void DisplayBox_PreviewKeyDown(object sender, PreviewKeyDownEventArgs e)
        {

        }
    }
}

public class Imaginary
{
    public double Real { get; set; }
    public double Imag { get; set; }

    ////////        Constructors

    public Imaginary()
    {
        this.Real = 0.0;
        this.Imag = 0.0;
    }

    public Imaginary(double real)
    {
        this.Real = real;
        this.Imag = 0.0;
    }

    public Imaginary(double real, double imag)
    {
        this.Real = real;
        this.Imag = imag;
    }

    ////////        Methods

    public Imaginary Conjugate()
    {
        return new Imaginary(this.Real, -this.Imag);
    }

    public double AbsoluteValueSquared()
    {
        return this.Real * this.Real + this.Imag * this.Imag;
    }

    ////////        Overloads

    ////    Static Overloads

    //  Addition

    public static Imaginary operator +(Imaginary inum1, Imaginary inum2)
    {
        return new Imaginary(inum1.Real + inum2.Real, inum1.Imag + inum2.Imag);
    }

    public static Imaginary operator +(Imaginary inum, double dnum)
    {
        return inum + new Imaginary(dnum);
    }

    public static Imaginary operator +(double dnum, Imaginary inum)
    {
        return new Imaginary(dnum) + inum;
    }

    //  Subtraction

    public static Imaginary operator -(Imaginary inum1, Imaginary inum2)
    {
        return new Imaginary(inum1.Real - inum2.Real, inum1.Imag - inum2.Imag);
    }

    public static Imaginary operator -(Imaginary inum, double dnum)
    {
        return inum - new Imaginary(dnum);
    }

    public static Imaginary operator -(double dnum, Imaginary inum)
    {
        return new Imaginary(dnum) - inum;
    }

    //  Multiplication

    public static Imaginary operator *(Imaginary inum1, Imaginary inum2)
    {
        double newReal = (inum1.Real * inum2.Real) - (inum1.Imag * inum2.Imag);
        double newImag = (inum1.Real * inum2.Imag) + (inum1.Imag * inum2.Real);
        return new Imaginary(newReal, newImag);
    }

    public static Imaginary operator *(Imaginary inum, double dnum)
    {
        return new Imaginary(inum.Real * dnum, inum.Imag);
    }

    public static Imaginary operator *(double dnum, Imaginary inum)
    {
        return inum * dnum;
    }

    // Division

    public static Imaginary operator /(Imaginary inum1, Imaginary inum2)
    {
        Imaginary numerator = inum1 * inum2.Conjugate();
        double denominator = inum2.AbsoluteValueSquared();
        return new Imaginary(numerator.Real / denominator, numerator.Imag / denominator);
    }

    public static Imaginary operator /(Imaginary inum, double dnum)
    {
        return new Imaginary(inum.Real / dnum, inum.Imag);
    }

    public static Imaginary operator /(double dnum, Imaginary inum)
    {
        return new Imaginary(dnum) / inum;
    }

    ////    Dynamic Overloads

    public override string ToString()
    {
        return PrintNumber(Real, Imag);
    }

    public string ToString(int roundLevel)
    {
        return PrintNumber(Math.Round(Real, roundLevel), Math.Round(Imag, roundLevel));
    }

    private string PrintNumber(double Re, double Im)
    {
        String connectingSymbol = " + ";
        if (Im == 0)
        {
            return Re.ToString();
        }
        else if (Re == 0)
        {
            if (Im == 1)
            {
                return "i";
            }
            else if (Im == -1)
            {
                return "-i";
            }
            else
            {
                return Im + "i";
            }
        }
        else if (Im < 0)
        {
            connectingSymbol = " - ";
            Im = Math.Abs(Im);
        }

        if (Im == 1)
        {
            return Re + connectingSymbol + "i";
        }
        else
        {
            return Re + connectingSymbol + Im + "i";
        }
    }

}

public static class Mandelbrot
{
    public static int CalculateJuliaFractal(Imaginary position, Imaginary subposition, int iters)
    {
        for(; iters > 0; iters--)
        {
            position = position * position + subposition; // The key formula right here
            if (position.AbsoluteValueSquared() > 4)
                break;
        }
        return iters;
    }

    public static int CalculateMandelbrotFractal(Imaginary position, int iters)
    {
        return Mandelbrot.CalculateJuliaFractal(position, position, iters);
    }

}

public class MandelRendererController
{
    private Imaginary LowerBound;
    private Imaginary UpperBound;
    private Point UpperPixelBound;
    private Imaginary Subposition;
    private MandelRenderer[] RenderObjects;
    private Thread[] Threads;
    private int PixelWidth;
    private uint ThreadCount;
    public Stopwatch Timer;

    public MandelRendererController(Imaginary lower, Imaginary upper, Point upperPixel, 
        Imaginary subposition, int iterations, uint threads)
    {
        if (threads == 0)
            throw new IndexOutOfRangeException("Must have at least 1 thread");
        ThreadCount = threads;
        LowerBound = lower;
        UpperBound = upper;
        UpperPixelBound = upperPixel;
        Subposition = subposition;
        Timer = new Stopwatch();

        RenderObjects = new MandelRenderer[ThreadCount];
        Point allocSize = new Point(UpperPixelBound.X / (int)ThreadCount + 1, UpperPixelBound.Y);
        double allocWidth = (upper.Real - lower.Real) / ThreadCount;
        PixelWidth = allocSize.X;
        Threads = new Thread[ThreadCount];
        for (int i=0; i < ThreadCount; i++)
        {
            Imaginary lowAllocBound = new Imaginary(LowerBound.Real + (allocWidth * i), LowerBound.Imag);
            Imaginary highAllocBound = new Imaginary(LowerBound.Real + (allocWidth * (i + 1)), UpperBound.Imag);
            RenderObjects[i] = new MandelRenderer(
                lowAllocBound, highAllocBound, allocSize, Subposition, iterations);
            Threads[i] = new Thread(RenderObjects[i].Render);
        }
    }

    public void Activate()
    {
        Timer.Start();
        for (int i=0; i < ThreadCount; i++)
            Threads[i].Start();
    }

    public void Compile(PictureBox p)
    {
        Graphics g = p.CreateGraphics();
        bool[] completed = new bool[ThreadCount];
        int numCompleted = 0;
        for (int i = 0; i < ThreadCount; i++)
            completed[i] = false;
        while (numCompleted < ThreadCount)
        {
            for (int i = 0; i < ThreadCount; i++)
            {
                if (completed[i])
                    continue;
                if (Threads[i].IsAlive)
                    continue;
                completed[i] = true;
                numCompleted++;
                PointF position = new PointF(i * PixelWidth, 0);
                Threads[i].Join();
                g.DrawImage(RenderObjects[i].Production, position);
            }
        }
        Timer.Stop();
    }

    protected class MandelRenderer
    {
        public Bitmap Production;
        private Imaginary LowerBound;
        private Imaginary UpperBound;
        private Point UpperPixelBound;
        private Imaginary Subposition;
        private int Iterations;

        public MandelRenderer(Imaginary lower, Imaginary upper, 
            Point upperPixel, Imaginary subposition, int iterations)
        {
            LowerBound = lower;
            UpperBound = upper;
            UpperPixelBound = upperPixel;
            Subposition = subposition;
            Iterations = iterations;
        }

        public void Render()
        {
            double Regulator = (double)Iterations / 255.0;
            Util.Ranges rutil = new Util.Ranges();
            Imaginary ipos = new Imaginary();
            Production = new Bitmap(UpperPixelBound.X, UpperPixelBound.Y);
            foreach (Util.Ranges.TandemRangeOutput xout in 
                rutil.SecondaryTandemRange(LowerBound.Real, UpperBound.Real, UpperPixelBound.X))
            {
                foreach (Util.Ranges.TandemRangeOutput yout in 
                    rutil.SecondaryTandemRange(LowerBound.Imag, UpperBound.Imag, UpperPixelBound.Y))
                {
                    ipos.Real = xout.RangeValue;
                    ipos.Imag = yout.RangeValue;
                    int x = xout.IterValue;
                    int y = yout.IterValue;
                    int calculation;
                    if (Subposition == null)
                        calculation = Mandelbrot.CalculateMandelbrotFractal(ipos, Iterations);
                    else
                        calculation = Mandelbrot.CalculateJuliaFractal(ipos, Subposition, Iterations);
                    int fval = (int) (calculation / Regulator);
                    Production.SetPixel(x, y, Color.FromArgb(fval, fval, fval));
                }
            }
        }
    }
}

namespace Util
{
    public class Ranges
    {

        public System.Collections.IEnumerator GetEnumerator()
        {
            yield return null;
        }

        public System.Collections.IEnumerable Range(double begin, double end, double increment)
        {
            for (double value = begin; value < end; value += increment)
                yield return value;
        }

        public System.Collections.IEnumerable Range(double begin, double end)
        {
            return Range(begin, end, 1);
        }
        public System.Collections.IEnumerable Range(double end)
        {
            return Range(0, end, 1);
        }

        public class TandemRangeOutput
        {
            public double RangeValue { get; }
            public int IterValue { get; }

            public TandemRangeOutput(double rval, int ival)
            {
                this.RangeValue = rval;
                this.IterValue = ival;
            }
            
            public override String ToString()
            {
                return String.Format("({0}, {1})", this.RangeValue, this.IterValue);
            }
        }

        public System.Collections.IEnumerable TandemRange(double begin, double end, double increment)
        {
            int iterator = 0;
            double value = begin;
            for (; value < end; value += increment, iterator++)
                yield return new TandemRangeOutput(value, iterator);
        }

        public System.Collections.IEnumerable TandemRange(double begin, double end)
        {
            return TandemRange(begin, end, 1);
        }

        public System.Collections.IEnumerable TandemRange(double end)
        {
            return TandemRange(0, end, 1);
        }

        public System.Collections.IEnumerable SecondaryTandemRange(double begin, double end, int peices)
        {
            double increment = (end - begin) / (double) peices;
            int iterator = 0;
            double value = begin;
            for (; iterator < peices; value += increment, iterator++)
                yield return new TandemRangeOutput(value, iterator);
        }
    }

    public static class Transform
    {
        public static double FromPixel(double minExtent, double maxExtent, int pixelHeight, int pixelTarget)
        {
            double range = maxExtent - minExtent;
            double percentage = ((double)pixelHeight) / ((double)pixelTarget);
            return percentage * range + minExtent;
        }

        public static int ToPixel(int pixelHeight, double minExtent, double maxExtent, double valueTarget)
        {
            double progression = valueTarget - minExtent;
            double range = maxExtent - minExtent;
            double percentage = progression / range;
            double approximation = percentage * pixelHeight;
            return (int)approximation;
        }
    }

    public class ProgressTracker
    {
        private double Value = 0;
        private int ProgressValue = 0; 

        private double MaxValue;
        private int MaxProgressValue;

        public ProgressTracker(double maxValue, int maxProgressValue)
        {
            MaxValue = maxValue;
            MaxProgressValue = maxProgressValue;
        }

        public ProgressTracker(double maxValue)
        {
            MaxValue = maxValue;
            MaxProgressValue = 100;
        }

        public int increment(double value)
        {
            Value += value;
            int newProgress = (int)(Value / MaxValue) * MaxProgressValue;
            int change = newProgress - ProgressValue;
            ProgressValue = newProgress;
            return change;
        }

    }
}