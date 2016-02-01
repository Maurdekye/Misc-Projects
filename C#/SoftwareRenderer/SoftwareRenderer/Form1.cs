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
        Raytracer OverheadCamera;
        Raytracer IsometricCamera;

        List<Triangle> TestTriangles;
        List<Triangle> Bunny;

        Thread RenderThread;

        public MainForm()
        {
            InitializeComponent();
            
            Bunny = Import("bunny.obj");
            TestTriangles = Import("testtris.obj");
        }

        private void RenderTarget_Paint(object sender, PaintEventArgs e)
        {
            OverheadCamera = new Raytracer(new Triangle(
                new Geometry.Point(0, 3, 0),
                new Geometry.Point(0, 2, 0),
                new Geometry.Point(1, 3, 0)
                ), RenderTarget);

            IsometricCamera = new Raytracer(new Triangle(
                new Geometry.Point(3, 3, -3),
                new Geometry.Point(2, 2, -2),
                new Geometry.Point(2, 4, -2)
                ), RenderTarget);

            if (RenderThread != null && RenderThread.IsAlive)
            {
                RenderThread.Abort();
                OverheadCamera.Clean();
            }
            RenderThread = new Thread(delegate ()
            {
                IsometricCamera.AsyncMultipleRender(TestTriangles, delegate(float Depth, Ray Raycast, Triangle Tri)
                {
                    int C = (int)(Depth * 32) % 256;
                    return Color.FromArgb(C, C, C);
                });
                IsometricCamera.AsyncDraw();
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
            return -Diff.Dot(A - P);
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
            return F >= 1;
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
            return new Point2D(A.X - B.X, A.Y - B.Y);
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
        public readonly uint ThreadCount = 24;

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
                for (int t = 0;t < ThreadCount;t++)
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
                    Geometry.Point a = Step.Extrapolate(PositionX);
                    yield return new PointPixelPair(Step.Extrapolate(PositionX), new Pixel(PixelX, PixelY));
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

        public abstract void Render(Triangle Tri, PixelShader Shader);

        public void AsyncMultipleRender(List<Triangle> Polygon)
        {
            AsyncMultipleRender(Polygon, delegate (float D, Ray R, Triangle T)
            {
                return Color.White;
            });
        }

        public abstract void AsyncMultipleRender(List<Triangle> Polygon, PixelShader Shader);

        // Embedded Structures

        public class PointPixelPair
        {
            public Geometry.Point Point { get; private set; }
            public Pixel Pixel { get; private set; }

            // Constructors

            public PointPixelPair(Geometry.Point Point, Pixel Pixel)
            {
                this.Point = Point;
                this.Pixel = Pixel;
            }

            public PointPixelPair()
            {
                this.Point = new Geometry.Point();
                this.Pixel = new Pixel();
            }

            // Overloads

            public override string ToString()
            {
                return String.Format("{0} | {1}", Pixel, Point);
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

            public GenericRenderThreadManager(Camera Parent, Pixel LowerCorner, BufferElement[,] ImageBuffer)
            {
                this.Parent = Parent;
                this.LowerCorner = LowerCorner;
                this.ImageBuffer = ImageBuffer;

                this.UpperCorner = new Pixel(LowerCorner.X + ImageBuffer.GetLength(0), LowerCorner.Y + ImageBuffer.GetLength(1));
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

            public abstract void RenderSingleTriangle(Triangle Tri, PixelShader Shader);

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
        // Constructors

        public Raytracer(Triangle Frame, PictureBox Target) : base(Frame, Target) { }
        
        internal override void InitializeRenderThreadManagers()
        {
            this.RenderThreadManagers = new RaytracerRenderThreadManager[ThreadCount];

            float Division = 0;
            for (int i = 0; i < ThreadCount; i++)
            {
                float LeftBounds = (float)Math.Floor(Division);
                Division += Target.Width / (float)ThreadCount;
                float RightBounds = (float)Math.Floor(Division);
                if (i == ThreadCount - 1)
                    RightBounds = Target.Width;
                int BufferWidth = (int)(RightBounds - LeftBounds);
                RenderThreadManagers[i] = new RaytracerRenderThreadManager(this, new Pixel((int)LeftBounds, 0), new BufferElement[BufferWidth, Target.Height]);
            }
        }

        // Utilities

        public override void Render(Triangle Tri, PixelShader Shader)
        {
            foreach (RaytracerRenderThreadManager ThreadManager in RenderThreadManagers)
            {
                ThreadManager.Render(Tri, Shader);
            }
        }

        public override void AsyncMultipleRender(List<Triangle> Polygon, PixelShader Shader)
        {
            foreach (RaytracerRenderThreadManager ThreadManager in RenderThreadManagers)
            {
                ThreadManager.AsyncMultipleRender(Polygon, Shader);
            }
        }

        // Embedded Structures

        public class RaytracerRenderThreadManager : GenericRenderThreadManager
        {
            // Constructors

            public RaytracerRenderThreadManager(Camera Parent, Pixel LowerCorner, BufferElement[,] ImageBuffer) : base(Parent, LowerCorner, ImageBuffer) { }

            // Utilities

            public override void RenderSingleTriangle(Triangle Tri, PixelShader Shader)
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
}
