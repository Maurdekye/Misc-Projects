using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using Render;
using Geometry;
using static Utility;
using System.Text.RegularExpressions;

namespace SoftwareRenderer
{
    public partial class MainForm : Form
    {
        Camera.PixelShader DefaultShader;

        List<Triangle> TestTriangles;
        List<Triangle> Bunny;

        Thread RenderThread;

        public MainForm()
        {
            InitializeComponent();
            
            Bunny = Import("bunny.obj");
            TestTriangles = Import("testtris.obj");

            DefaultShader = delegate (float Depth, Ray Raycast, Triangle Tri)
            {
                int C = (int)(Depth * 32) % 256;
                return Color.FromArgb(C, C, C);
            };
        }

        private void RenderTarget_Paint(object sender, PaintEventArgs e)
        {
            Raytracer OverheadCamera = new Raytracer(new Triangle(
                new Geometry.Point(0, 3, 0),
                new Geometry.Point(0, 2, 0),
                new Geometry.Point(1, 3, 0)
                ), RenderTarget);

            Raytracer IsometricCamera = new Raytracer(new Triangle(
                new Geometry.Point(3, 3, -3),
                new Geometry.Point(2, 2, -2),
                new Geometry.Point(2, 4, -2)
                ), RenderTarget);

            Rasterizer RasterCam = new Rasterizer(new Triangle(
                new Geometry.Point(0, 3, 0),
                new Geometry.Point(0, 2, 0),
                new Geometry.Point(1, 3, 0)
                ), RenderTarget);

            if (RenderThread != null && RenderThread.IsAlive)
            {
                RenderThread.Abort();
                OverheadCamera.Clean();
            }

            RenderThread = new Thread(delegate ()
            {
                RasterCam.AsyncMultipleRender(TestTriangles, DefaultShader);
                RasterCam.AsyncDraw();
            });
            RenderThread.Start();
        }
    }
}

namespace Geometry
{

    public class Point
    {
        public float X, Y, Z;

        // Constructors

        public Point(float X, float Y, float Z)
        {
            this.X = X;
            this.Y = Y;
            this.Z = Z;
        }

        public Point(float F)
        {
            this.X = F;
            this.Y = F;
            this.Z = F;
        }

        public Point()
        {
            this.X = 0;
            this.Y = 0;
            this.Z = 0;
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", X, Y, Z);
        }

        public static Point operator -(Point A, Point B)
        {
            return new Point(A.X - B.X, A.Y - B.Y, A.Z - B.Z);
        }

        public static Point operator -(Point P, float F)
        {
            return P - new Point(F);
        }

        public static Point operator +(Point A, Point B)
        {
            return new Point(A.X + B.X, A.Y + B.Y, A.Z + B.Z);
        }

        public static Point operator +(Point P, float F)
        {
            return P + new Point(F);
        }

        public static Point operator *(Point P, float F)
        {
            return new Point(P.X * F, P.Y * F, P.Z * F);
        }

        public static Point operator /(Point P, float F)
        {
            return new Point(P.X / F, P.Y / F, P.Z / F);
        }

        // Utilities

        public float AbsoluteValue()
        {
            return (float)Math.Sqrt(X * X + Y * Y + Z * Z);
        }

        public float Distance(Point P)
        {
            return (P - this).AbsoluteValue();
        }

        public float Dot(Point P)
        {
            return X * P.X + Y * P.Y + Z * P.Z;
        }

        public Point Cross(Point P)
        {
            Point Result = new Point();
            Result.X = Y * P.Z - Z * P.Y;
            Result.Y = Z * P.X - X * P.Z;
            Result.Z = X * P.Y - Y * P.X;
            return Result;
        }

        public Point Normalized()
        {
            float Abs = AbsoluteValue();
            if (Abs == 0)
            {
                return new Point(0, 1, 0);
            }
            else
            {
                return this * (1f / Abs);
            }
        }
    }

    public class Segment
    {

        public Point A, B;

        // Constructors

        public Segment(Point A, Point B)
        {
            this.A = A;
            this.B = B;
        }

        public Segment(Point P)
        {
            this.A = new Point();
            this.B = P;
        }

        public Segment()
        {
            this.A = new Point();
            this.B = new Point();
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("{0}={1}", A, B);
        }

        // Self Utilities

        public Point Differernce()
        {
            return B - A;
        }

        public float Length()
        {
            return Differernce().AbsoluteValue();
        }

        // Scalar Utilities

        public Point Extrapolate(float F)
        {
            return A + Differernce() * F;
        }

        public virtual bool IsUnextrapolationInLine(float F)
        {
            return F >= 0 && F <= 1;
        }

        // Point Utilities

        public float ClosestPointUnextrapolated(Point P)
        {
            Point Diff = Differernce();
            float Denominator = Diff.Dot(Diff);
            if (Denominator == 0)
                return 0;
            return -Diff.Dot(A - P) / Denominator;
        }

        public Point ClosestPoint(Point P)
        {
            return Extrapolate(ClosestPointUnextrapolated(P));
        }

        // Triangle Utilities

        public Maybe<float> UnextrapolatedUncontainedIntersection(Triangle T)
        {
            Point Normal = T.Normal();
            float Denominator = Normal.Dot(Differernce());
            if (Denominator == 0)
                return new Maybe<float>();
            return new Maybe<float>(-Normal.Dot(A - T.A) / Denominator);
        }

        public bool IsUnextrapolationInTriangle(Maybe<float> F, Triangle Tri)
        {
            if (F.IsNull)
                return false;
            return IsUnextrapolationInTriangle(F.Value(), Tri);
        }
        public bool IsUnextrapolationInTriangle(float F, Triangle Tri)
        {
            Point U = Tri.B - Tri.A;
            Point V = Tri.C - Tri.A;
            Point W = Extrapolate(F) - Tri.A;
            float UV = U.Dot(V);
            float UU = U.Dot(U);
            float WV = W.Dot(V);
            float WU = W.Dot(U);
            float VV = V.Dot(V);
            float Denominator = UV * UV - UU * VV;
            if (Denominator == 0)
                return false;

            float S = (UV * WV - VV * WU) / Denominator;
            if (S < 0 || S > 1)
                return false;

            float T = (UV * WU - UU * WV) / Denominator;
            if (T < 0 || S + T > 1)
                return false;

            return true;
        }
    }

    public class Ray : Segment
    {
        public Ray(Point A, Point B) : base(A, B) { }

        // Overrides

        public override string ToString()
        {
            return String.Format("{0}>{1}", A, B);
        }

        public override bool IsUnextrapolationInLine(float F)
        {
            return F >= 0;
        }
    }

    public class Triangle
    {
        public Point A, B, C;

        // Constructors

        public Triangle(Point A, Point B, Point C)
        {
            this.A = A;
            this.B = B;
            this.C = C;
        }

        public Triangle()
        {
            this.A = new Point();
            this.B = new Point();
            this.C = new Point();
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("{{0}:{1}:{2}}", A, B, C);
        }

        // Utilities

        public Point Normal()
        {
            Point V = C - A;
            Point W = B - A;
            return W.Cross(V).Normalized();
        }

        public float Distance(Point P)
        {
            return Normal().Dot(P - A);
        }
    }

    public class Point2D
    {
        public float X, Y;

        // Constructors

        public Point2D(float X, float Y)
        {
            this.X = X;
            this.Y = Y;
        }

        public Point2D(float F)
        {
            this.X = F;
            this.Y = F;
        }

        public Point2D()
        {
            this.X = 0;
            this.Y = 0;
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("({0}, {1})", X, Y);
        }

        public static Point2D operator -(Point2D A, Point2D B)
        {
            return new Point2D(A.X - B.X, A.Y - B.Y);
        }

        public static Point2D operator -(Point2D P, float F)
        {
            return P - new Point2D(F);
        }

        public static Point2D operator +(Point2D A, Point2D B)
        {
            return new Point2D(A.X + B.X, A.Y + B.Y);
        }

        public static Point2D operator +(Point2D P, float F)
        {
            return P + new Point2D(F);
        }

        public static Point2D operator *(Point2D P, float F)
        {
            return new Point2D(P.X * F, P.Y * F);
        }

        public static Point2D operator /(Point2D P, float F)
        {
            return new Point2D(P.X / F, P.Y / F);
        }

        // Utilities

        public float Cross(Point2D P)
        {
            return (X * P.Y) - (Y * P.X);
        }

        public Pixel Pixel()
        {
            return new Utility.Pixel((int)Math.Round(X), (int)Math.Round(Y));
        }
    }

    public class Segment2D
    {
        public Point2D A, B;
        public Segment OldSegment;

        private Point2D LowerCorner;
        private Point2D UpperCorner;

        // Constructors

        public Segment2D(Point2D A, Point2D B, Segment OldSegment)
        {
            Init(A, B, OldSegment);
        }

        public Segment2D(Point2D A, Point2D B)
        {
            Init(A, B, new Segment());
        }

        public Segment2D()
        {
            Init(new Point2D(), new Point2D(), new Segment());
        }

        public void Init(Point2D A, Point2D B, Segment OldSegment)
        {
            this.A = A;
            this.B = B;
            this.OldSegment = OldSegment;

            this.LowerCorner = new Point2D(Math.Max(A.X, B.X), Math.Max(A.Y, B.Y));
            this.UpperCorner = new Point2D(Math.Min(A.X, B.X), Math.Min(A.Y, B.Y));
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("[{0}-{1}]", A, B);
        }

        // Utilities

        public Point2D Difference()
        {
            return B - A;
        }

        public float Side(Point2D P)
        {
            return (B.X - A.X) * (P.Y - A.Y) - (B.Y - A.Y) * (P.X - A.X);
        }

        public bool IsUnextrapolationOnLine(float F)
        {
            return F >= 0 && F <= 1;
        }

        public Maybe<float> UnextrapolatedIntersection(Segment2D S)
        {
            Point2D Numerator = S.A - A;
            float Denominator = Difference().Cross(S.Difference());
            if (Denominator == 0)
                return new Maybe<float>();
            float T = Numerator.Cross(S.Difference()) / Denominator;
            if (!IsUnextrapolationOnLine(T))
                return new Maybe<float>();
            float U = Numerator.Cross(Difference()) / Denominator;
            if (!S.IsUnextrapolationOnLine(U))
                return new Maybe<float>();
            return new Maybe<float>(T);
        }

        public Point2D Extrapolate(float F)
        {
            return A + Difference() * F;
        }
    }

    public class Triangle2D
    {
        public Point2D A, B, C;
        public Triangle OldTriangle;

        public Point2D UpperCorner, LowerCorner;
        public Segment2D SegmentAB, SegmentBC, SegmentCA;
        public Point2D Dimensions;

        // Constructors

        public Triangle2D(Point2D A, Point2D B, Point2D C, Triangle OldTriangle)
        {
            Init(A, B, C, OldTriangle);
        }

        public Triangle2D(Point2D A, Point2D B, Point2D C)
        {
            Init(A, B, C, new Triangle());
        }

        public Triangle2D()
        {
            Init(new Point2D(), new Point2D(), new Point2D(), new Triangle());
        }

        public void Init(Point2D A, Point2D B, Point2D C, Triangle OldTriangle)
        {
            this.A = A;
            this.B = B;
            this.C = C;
            this.OldTriangle = OldTriangle;

            this.SegmentAB = new Segment2D(A, B);
            this.SegmentBC = new Segment2D(B, C);
            this.SegmentCA = new Segment2D(C, A);

            this.UpperCorner = new Point2D();
            this.UpperCorner.X = Math.Max(A.X, Math.Max(B.X, C.X));
            this.UpperCorner.Y = Math.Max(A.Y, Math.Max(B.Y, C.Y));
            this.LowerCorner = new Point2D();
            this.LowerCorner.X = Math.Min(A.X, Math.Min(B.X, C.X));
            this.LowerCorner.Y = Math.Min(A.Y, Math.Min(B.Y, C.Y));

            this.Dimensions = new Point2D(UpperCorner.X - LowerCorner.X, UpperCorner.Y - LowerCorner.Y);
        }

        // Overrides 

        public override string ToString()
        {
            return String.Format("{{0}.{1}.{2}}", A, B, C);
        }

        // Utilities

        public bool Contains(Point2D P)
        {
            return SegmentAB.Side(P) >= 0 && SegmentBC.Side(P) >= 0 && SegmentCA.Side(P) >= 0;
        }
    }

}

public static class Utility
{
    public class Maybe<T>
    {
        private T ActualValue;
        public bool IsNull { get; private set; }

        // Constructors

        public Maybe(T ActualValue)
        {
            this.ActualValue = ActualValue;
            this.IsNull = false;
        }

        public Maybe()
        {
            this.ActualValue = default(T);
            this.IsNull = true;
        }

        // Overrides

        public override string ToString()
        {
            if (IsNull)
                return "Nothing";
            return "Just " + ActualValue.ToString();
        }

        // Utilities

        public T Value()
        {
            if (IsNull)
                throw new Exception("Value does not exist");
            return ActualValue;
        }

    }

    public class Pixel
    {
        public int X, Y;

        // Constructors

        public Pixel(int X, int Y)
        {
            this.X = X;
            this.Y = Y;
        }

        public Pixel()
        {
            this.X = 0;
            this.Y = 0;
        }

        public Pixel(Pixel P)
        {
            this.X = P.X;
            this.Y = P.Y;
        }

        // Overrides

        public override string ToString()
        {
            return String.Format("[{0}, {1}]", X, Y);
        }

        // Utilities

        public PointF PointF()
        {
            return new PointF((float)X, (float)Y);
        }
    }

    public static List<Triangle> Import(string Filepath)
    {
        List<Triangle> Polygon = new List<Triangle>();
        if (Filepath.EndsWith(".obj"))
        {
            List<Geometry.Point> Verticies = new List<Geometry.Point>();
            foreach (string Line in System.IO.File.ReadLines(Filepath))
            {
                if (Line.StartsWith("#"))
                    continue;
                else if (Line.StartsWith("v"))
                {
                    List<Match> Matches = Regex.Matches(Line, @"\s+?\S+").Cast<Match>().ToList();
                    if (Matches.Count < 3)
                        continue;
                    float X = float.Parse(Matches[0].ToString());
                    float Y = float.Parse(Matches[1].ToString());
                    float Z = float.Parse(Matches[2].ToString());
                    Verticies.Add(new Geometry.Point(X, Y, Z));
                }
                else if (Line.StartsWith("f"))
                {
                    List<Match> Matches = Regex.Matches(Line, @"\s+?\S+").Cast<Match>().ToList();
                    if (Matches.Count < 3)
                        continue;
                    int X = int.Parse(Matches[0].ToString());
                    int Y = int.Parse(Matches[1].ToString());
                    int Z = int.Parse(Matches[2].ToString());
                    Polygon.Add(new Triangle(Verticies[X-1], Verticies[Y-1], Verticies[Z-1]));
                }
            }
        }
        else
        {
            throw new NotImplementedException("Unsupported file type");
        }
        return Polygon;
    }

}

namespace Render
{
    public abstract class Camera
    {
        public readonly uint ThreadCount = 64;

        public Triangle Frame;
        public PictureBox Target;

        internal float Width, Height;
        internal float Ratio;
        internal Geometry.Point UpperRightCorner, UpperLeftCorner, LowerRightCorner, LowerLeftCorner;

        internal BufferElement[,] ImageBuffer;
        internal GenericRenderThreadManager[] RenderThreadManagers;
        public delegate Color PixelShader(float Depth, Ray Raycast, Triangle Tri);

        // Constructors

        public Camera(Triangle Frame, PictureBox Target)
        {
            Init(Frame, Target);
        }

        public Camera()
        {
            Init(new Triangle(), new PictureBox());
        }

        public void Init(Triangle Frame, PictureBox Target)
        {
            this.Frame = Frame;
            this.Target = Target;

            this.ImageBuffer = new BufferElement[Target.Width, Target.Height];

            Segment VerticalSegment = new Segment(Frame.A, Frame.C);

            this.Height = VerticalSegment.Length() * 2;
            this.Ratio = (float)Target.Width / Target.Height;
            this.Width = Height * Ratio;

            Geometry.Point RightSideMidpoint = Frame.A + Frame.Normal() * Width;
            Geometry.Point LeftSideMidpoint = Frame.A - Frame.Normal() * Width;
            Geometry.Point VerticalDifference = VerticalSegment.Differernce();
            this.UpperRightCorner = RightSideMidpoint + VerticalDifference;
            this.LowerRightCorner = RightSideMidpoint - VerticalDifference;
            this.UpperLeftCorner = LeftSideMidpoint + VerticalDifference;
            this.LowerLeftCorner = LeftSideMidpoint - VerticalDifference;

            InitializeRenderThreadManagers();

            Clean();
        }

        internal abstract void InitializeRenderThreadManagers();

        // Utilities

        internal void Wait()
        {
            uint Running = ThreadCount;
            bool[] Finished = new bool[ThreadCount];
            for (int i = 0; i < ThreadCount; i++)
                Finished[i] = false;
            while (Running > 0)
            {
                for (int t = 0; t < ThreadCount; t++)
                {
                    if (RenderThreadManagers[t].Finished() && !Finished[t])
                    {
                        Running--;
                        Finished[t] = true;
                    }
                }
            }
        }

        public void Compile()
        {
            Wait();
            foreach (GenericRenderThreadManager Thread in RenderThreadManagers)
            {
                for (int x = 0; x < Thread.ImageBuffer.GetLength(0); x++)
                {
                    for (int y = 0; y < Thread.ImageBuffer.GetLength(1); y++)
                    {
                        ImageBuffer[x + Thread.LowerCorner.X, y + Thread.LowerCorner.Y] = Thread.ImageBuffer[x, y];
                    }
                }
            }
        }

        public Bitmap CreateBitmap()
        {
            Compile();
            Bitmap Image = new Bitmap(Target.Width, Target.Height);
            for (int x = 0; x < Target.Width; x++)
            {
                for (int y = 0; y < Target.Height; y++)
                {
                    if (ImageBuffer[x, y] == null)
                    {
                        Image.SetPixel(x, y, Color.Black);
                    }
                    else
                    {
                        Image.SetPixel(x, y, ImageBuffer[x, y].VisualElement);
                    }
                }
            }
            return Image;
        }

        public void Draw()
        {
            Bitmap Production = CreateBitmap();
            Target.CreateGraphics().DrawImage(Production, new PointF(0, 0));
        }

        public void AsyncDraw()
        {
            Graphics G = Target.CreateGraphics();
            uint Running = ThreadCount;
            bool[] Finished = new bool[ThreadCount];
            for (int i = 0; i < ThreadCount; i++)
                Finished[i] = false;
            while (Running > 0)
            {
                for (int t = 0; t < ThreadCount; t++)
                {
                    if (RenderThreadManagers[t].Finished() && !Finished[t])
                    {
                        Running--;
                        Finished[t] = true;
                        Bitmap Slice = new Bitmap(RenderThreadManagers[t].ImageBuffer.GetLength(0), RenderThreadManagers[t].ImageBuffer.GetLength(1));
                        for (int x = 0; x < RenderThreadManagers[t].ImageBuffer.GetLength(0); x++)
                        {
                            for (int y = 0; y < RenderThreadManagers[t].ImageBuffer.GetLength(1); y++)
                            {
                                if (RenderThreadManagers[t].ImageBuffer[x, y] == null)
                                    Slice.SetPixel(x, y, Color.Black);
                                else
                                    Slice.SetPixel(x, y, RenderThreadManagers[t].ImageBuffer[x, y].VisualElement);
                            }
                        }
                        G.DrawImage(Slice, RenderThreadManagers[t].LowerCorner.PointF());
                    }
                }
            }
        }

        public void Clean()
        {
            for (int x = 0; x < Target.Width; x++)
            {
                for (int y = 0; y < Target.Height; y++)
                {
                    ImageBuffer[x, y] = new BufferElement(Color.Goldenrod);
                }
            }
            foreach (GenericRenderThreadManager GRTM in RenderThreadManagers)
            {
                for (int x = 0; x < GRTM.ImageBuffer.GetLength(0); x++)
                {
                    for (int y = 0; y < GRTM.ImageBuffer.GetLength(1); y++)
                    {
                        GRTM.ImageBuffer[x, y] = new BufferElement();
                    }
                }
            }
        }
        
        public IEnumerable<PointPixelPair> GetPixels(Pixel LowerBounds, Pixel UpperBounds)
        {
            Segment LeftRail = new Segment(LowerLeftCorner, UpperLeftCorner);
            Segment RightRail = new Segment(LowerRightCorner, UpperRightCorner);
            float PositionY = LowerBounds.Y / (float)Target.Height;
            for (int PixelY = LowerBounds.Y; PixelY <= UpperBounds.Y; PixelY += 1, PositionY += 1f / Target.Height)
            {
                Segment Step = new Segment(RightRail.Extrapolate(PositionY), LeftRail.Extrapolate(PositionY));
                float PositionX = LowerBounds.X / (float)Target.Width;
                for (int PixelX = LowerBounds.X; PixelX <= UpperBounds.X; PixelX += 1, PositionX += 1f / Target.Width)
                {
                    yield return new PointPixelPair(Step.Extrapolate(PositionX), new Point2D(PositionX, PositionY), new Pixel(PixelX, PixelY));
                }
            }
        }

        public void Render(Triangle Tri)
        {
            Render(Tri, delegate (float D, Ray R, Triangle T)
            {
                return Color.White;
            });
        }
        
        public void AsyncMultipleRender(List<Triangle> Polygon)
        {
            AsyncMultipleRender(Polygon, delegate (float D, Ray R, Triangle T)
            {
                return Color.White;
            });
        }
        public void Render(Triangle Tri, PixelShader Shader)
        {
            foreach (GenericRenderThreadManager ThreadManager in RenderThreadManagers)
            {
                ThreadManager.Render(Tri, Shader);
            }
        }

        public void AsyncMultipleRender(List<Triangle> Polygon, PixelShader Shader)
        {
            foreach (GenericRenderThreadManager ThreadManager in RenderThreadManagers)
            {
                ThreadManager.AsyncMultipleRender(Polygon, Shader);
            }
        }

        public void DebugThreadlessRender(Triangle Tri, PixelShader Shader)
        {
            foreach (GenericRenderThreadManager ThreadManager in RenderThreadManagers)
            {
                ThreadManager.RenderSingleTriangle(Tri, Shader);
                ThreadManager.RenderThread = new Thread(delegate () { });
                ThreadManager.RenderThread.Start();
            }
        }

        // Embedded Structures

        public class PointPixelPair
        {
            public Geometry.Point Point { get; private set; }
            public Point2D Point2D { get; private set; }
            public Pixel Pixel { get; private set; }

            // Constructors

            public PointPixelPair(Geometry.Point Point, Point2D Point2D, Pixel Pixel)
            {
                this.Point = Point;
                this.Pixel = Pixel;
                this.Point2D = Point2D;
            }

            public PointPixelPair()
            {
                this.Point = new Geometry.Point();
                this.Pixel = new Pixel();
                this.Point2D = new Point2D();
            }

            // Overloads

            public override string ToString()
            {
                return String.Format("{0}:{1} | {2}", Pixel, Point2D, Point);
            }
        }

        public class BufferElement
        {
            public Color VisualElement;
            public float ZElement;

            // Constructors

            public BufferElement(Color VisualElement, float ZElement)
            {
                this.VisualElement = VisualElement;
                this.ZElement = ZElement;
            }

            public BufferElement(Color VisualElement)
            {
                this.VisualElement = VisualElement;
                this.ZElement = -1;
            }

            public BufferElement()
            {
                this.VisualElement = Color.Black;
                this.ZElement = -1;
            }
        }

        public abstract class GenericRenderThreadManager
        {
            public Camera Parent;
            public Pixel LowerCorner, UpperCorner;
            public volatile BufferElement[,] ImageBuffer;
            internal Thread RenderThread;

            // Constructors

            public GenericRenderThreadManager(Camera Parent, Pixel LowerCorner, Pixel UpperCorner)
            {
                this.Parent = Parent;
                this.LowerCorner = LowerCorner;
                this.UpperCorner = UpperCorner;

                this.ImageBuffer = new BufferElement[UpperCorner.X - LowerCorner.X, UpperCorner.Y - LowerCorner.Y];
            }

            public GenericRenderThreadManager()
            {
                this.Parent = null;
                this.LowerCorner = new Pixel();
                this.UpperCorner = new Pixel();
                this.ImageBuffer = new BufferElement[1, 1];
            }

            // Utilities

            internal void Impress(Pixel Pixel, BufferElement Elem)
            {
                if (Pixel.X >= UpperCorner.X || Pixel.Y >= UpperCorner.Y)
                    return;
                if (Pixel.X < LowerCorner.X || Pixel.Y < LowerCorner.Y)
                    return;
                BufferElement CurrentBufferElement = ImageBuffer[Pixel.X - LowerCorner.X, Pixel.Y - LowerCorner.Y];
                if (CurrentBufferElement == null)
                {
                    ImageBuffer[Pixel.X - LowerCorner.X, Pixel.Y - LowerCorner.Y] = Elem;
                    return;
                }
                float OldZElement = CurrentBufferElement.ZElement;
                if (Elem.ZElement < 0 || OldZElement < 0 || Elem.ZElement < OldZElement)
                    ImageBuffer[Pixel.X - LowerCorner.X, Pixel.Y - LowerCorner.Y] = Elem;
            }

            public bool Finished()
            {
                return !RenderThread.IsAlive;
            }

            internal abstract void RenderSingleTriangle(Triangle Tri, PixelShader Shader);

            public void Render(Triangle Tri)
            {
                Render(Tri, delegate (float D, Ray R, Triangle T)
                {
                    return Color.White;
                });
            }

            public void Render(Triangle Tri, PixelShader Shader)
            {
                RenderThread = new Thread(delegate () { RenderSingleTriangle(Tri, Shader); });
                RenderThread.Start();
            }

            public void AsyncMultipleRender(List<Triangle> Polygon)
            {
                AsyncMultipleRender(Polygon, delegate (float D, Ray R, Triangle T)
                {
                    return Color.White;
                });
            }

            public void AsyncMultipleRender(List<Triangle> Polygon, PixelShader Shader)
            {
                RenderThread = new Thread(delegate ()
                {
                    foreach (Triangle Tri in Polygon)
                        RenderSingleTriangle(Tri, Shader);
                });
                RenderThread.Start();
            }
        }
    }

    public class Raytracer : Camera
    {
        public Raytracer(Triangle Frame, PictureBox Target) : base(Frame, Target) { }

        internal override void InitializeRenderThreadManagers()
        {
            this.RenderThreadManagers = new RaytracerRenderThreadManager[ThreadCount];

            float Division = 0;
            for (int i = 0; i < ThreadCount; i++)
            {
                float LeftBounds = (float)Math.Round(Division);
                Division += Target.Width / (float)ThreadCount;
                float RightBounds = (float)Math.Round(Division);
                RenderThreadManagers[i] = new RaytracerRenderThreadManager(this, new Pixel((int)LeftBounds, 0), new Pixel((int)RightBounds, Target.Height));
            }
        }

        internal class RaytracerRenderThreadManager : GenericRenderThreadManager
        {
            public RaytracerRenderThreadManager(Camera Parent, Pixel LowerCorner, Pixel UpperCorner) : base(Parent, LowerCorner, UpperCorner) { }

            internal override void RenderSingleTriangle(Triangle Tri, PixelShader Shader)
            {
                foreach (PointPixelPair Pair in Parent.GetPixels(LowerCorner, UpperCorner))
                {
                    Ray R = new Ray(Pair.Point, Parent.Frame.B);
                    Maybe<float> MaybeDepth = R.UnextrapolatedUncontainedIntersection(Tri);
                    if (MaybeDepth.IsNull)
                        continue;
                    float Depth = MaybeDepth.Value();
                    if (!R.IsUnextrapolationInLine(Depth))
                        continue;
                    if (!R.IsUnextrapolationInTriangle(Depth, Tri))
                        continue;
                    Impress(Pair.Pixel, new BufferElement(Shader(Depth, R, Tri), Depth));
                }
            }
        }

    }

    public class Rasterizer : Camera
    {
        public Rasterizer(Triangle Frame, PictureBox Target) : base(Frame, Target) { }

        internal override void InitializeRenderThreadManagers()
        {
            this.RenderThreadManagers = new RasterizerRenderThreadManager[ThreadCount];

            int GridHeight = (int)Math.Sqrt(ThreadCount);
            while (GridHeight > 1 && ThreadCount % GridHeight != 0)
                GridHeight--;
            int GridWidth = (int)ThreadCount / GridHeight;
            Point2D BoxSize = new Point2D((float)Target.Width / GridWidth, (float)Target.Height / GridHeight);
            Point2D Division = new Point2D();
            int t = 0;
            for (int x = 0; x < GridWidth; x++)
            {
                Division.Y = 0;
                for (int y = 0; y < GridHeight; y++)
                {
                    Pixel LowerBounds = Division.Pixel();
                    Division += BoxSize;
                    Pixel UpperBounds = Division.Pixel();
                    Division.X -= BoxSize.X;
                    RenderThreadManagers[t] = new RasterizerRenderThreadManager(this, LowerBounds, UpperBounds);
                    t++;
                }
                Division.X += BoxSize.X;
            }
        }

        internal class RasterizerRenderThreadManager : GenericRenderThreadManager
        {
            private Triangle ScreenTri;
            private Segment VerticalMargin, HorizontalMargin;

            public RasterizerRenderThreadManager(Camera Parent, Pixel LowerCorner, Pixel UpperCorner) : base(Parent, LowerCorner, UpperCorner)
            {
                this.ScreenTri = new Triangle(Parent.UpperLeftCorner, Parent.UpperRightCorner, Parent.LowerLeftCorner);
                this.VerticalMargin = new Segment(Parent.LowerLeftCorner, Parent.UpperLeftCorner);
                this.HorizontalMargin = new Segment(Parent.UpperLeftCorner, Parent.UpperRightCorner);
            }
            
            internal override void RenderSingleTriangle(Triangle Tri, PixelShader Shader)
            {
                Ray PointARayCast = new Ray(Tri.A, Parent.Frame.B);
                Ray PointBRayCast = new Ray(Tri.B, Parent.Frame.B);
                Ray PointCRayCast = new Ray(Tri.C, Parent.Frame.B);

                Maybe<float> PossiblePointAIntersection = PointARayCast.UnextrapolatedUncontainedIntersection(ScreenTri);
                if (PossiblePointAIntersection.IsNull) return;
                Geometry.Point PointAScreenIntersection = PointARayCast.Extrapolate(PossiblePointAIntersection.Value());

                Maybe<float> PossiblePointBIntersection = PointBRayCast.UnextrapolatedUncontainedIntersection(ScreenTri);
                if (PossiblePointBIntersection.IsNull) return;
                Geometry.Point PointBScreenIntersection = PointBRayCast.Extrapolate(PossiblePointBIntersection.Value());

                Maybe<float> PossiblePointCIntersection = PointCRayCast.UnextrapolatedUncontainedIntersection(ScreenTri);
                if (PossiblePointCIntersection.IsNull) return;
                Geometry.Point PointCScreenIntersection = PointCRayCast.Extrapolate(PossiblePointCIntersection.Value());

                Point2D PointA2DPosition = new Point2D(
                    HorizontalMargin.ClosestPointUnextrapolated(PointAScreenIntersection),
                    VerticalMargin.ClosestPointUnextrapolated(PointAScreenIntersection)
                    );
                if (PointA2DPosition.X < 0 || PointA2DPosition.Y < 0 || PointA2DPosition.X > 1 || PointA2DPosition.Y > 1) return;

                Point2D PointB2DPosition = new Point2D(
                    HorizontalMargin.ClosestPointUnextrapolated(PointBScreenIntersection),
                    VerticalMargin.ClosestPointUnextrapolated(PointBScreenIntersection)
                    );
                if (PointB2DPosition.X < 0 || PointB2DPosition.Y < 0 || PointB2DPosition.X > 1 || PointB2DPosition.Y > 1) return;

                Point2D PointC2DPosition = new Point2D(
                    HorizontalMargin.ClosestPointUnextrapolated(PointCScreenIntersection),
                    VerticalMargin.ClosestPointUnextrapolated(PointCScreenIntersection)
                    );
                if (PointC2DPosition.X < 0 || PointC2DPosition.Y < 0 || PointC2DPosition.X > 1 || PointC2DPosition.Y > 1) return;

                Triangle2D ProjectedTriangle = new Triangle2D(
                    PointA2DPosition,
                    PointB2DPosition,
                    PointC2DPosition
                    );

                Point2D UpperBoundBoxCorner = new Point2D(
                    Math.Max(PointA2DPosition.X, Math.Max(PointB2DPosition.X, PointC2DPosition.X)),
                    Math.Max(PointA2DPosition.Y, Math.Max(PointB2DPosition.Y, PointC2DPosition.Y))
                    );

                Point2D LowerBoundBoxCorner = new Point2D(
                    Math.Min(PointA2DPosition.X, Math.Min(PointB2DPosition.X, PointC2DPosition.X)),
                    Math.Min(PointA2DPosition.Y, Math.Min(PointB2DPosition.Y, PointC2DPosition.Y))
                    );

                Pixel UpperPixel = new Pixel(
                    (int)Math.Round(UpperBoundBoxCorner.X * Parent.Target.Width),
                    (int)Math.Round(UpperBoundBoxCorner.Y * Parent.Target.Height)
                    );

                Pixel LowerPixel = new Pixel(
                    (int)Math.Round(LowerBoundBoxCorner.X * Parent.Target.Width),
                    (int)Math.Round(LowerBoundBoxCorner.Y * Parent.Target.Height)
                    );

                if (UpperPixel.X < LowerCorner.X) return;
                if (UpperPixel.Y < LowerCorner.Y) return;
                if (LowerPixel.X > UpperCorner.X) return;
                if (LowerPixel.Y > UpperCorner.Y) return;

                UpperPixel.X = Math.Max(LowerCorner.X, Math.Min(UpperPixel.X, UpperCorner.X));
                UpperPixel.Y = Math.Max(LowerCorner.Y, Math.Min(UpperPixel.Y, UpperCorner.Y));
                LowerPixel.X = Math.Max(LowerCorner.X, Math.Min(LowerPixel.X, UpperCorner.X));
                LowerPixel.Y = Math.Max(LowerCorner.Y, Math.Min(LowerPixel.Y, UpperCorner.Y));

                foreach (PointPixelPair Pair in Parent.GetPixels(LowerPixel, UpperPixel))
                {
                    if (ProjectedTriangle.Contains(Pair.Point2D))
                    {
                        Ray Raycast = new Ray(Pair.Point, Parent.Frame.B);
                        Maybe<float> Depth = Raycast.UnextrapolatedUncontainedIntersection(Tri);
                        if (Depth.IsNull) continue;
                        Impress(Pair.Pixel, new BufferElement(Shader(Depth.Value(), Raycast, Tri), Depth.Value()));
                    }
                }
            }
        }
    }
}
