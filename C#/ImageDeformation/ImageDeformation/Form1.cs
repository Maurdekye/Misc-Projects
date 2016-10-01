using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ImageDeformation
{
    public partial class Form1 : Form
    {
        Bitmap image;

        public Form1()
        {
            Bitmap stockimage = (Bitmap)Image.FromFile("testim.bmp");
            double xres = stockimage.Width;
            double yres = stockimage.Height;
            DeformationMatrix Matrix = new DeformationMatrix(stockimage);
            Matrix.ApplyTransformation(delegate (Node n)
            {
                Node In = new Node(n.X / xres - 0.5, n.Y / yres - 0.5);
                Node Out = new Node(In);

                /*double theta = Math.PI / 8.0;
                Out.X = Math.Cos(theta) * In.X - Math.Sin(theta) * In.Y;
                Out.Y = Math.Sin(theta) * In.X + Math.Cos(theta) * In.Y;*/

                /*Out.X = In.X * In.X * In.X * 4;
                Out.Y = In.Y * In.Y * In.Y * 4;*/



                return new Node((Out.X + 0.5) * xres, (Out.Y + 0.5) * yres);
            });
            image = Matrix.ToBitmap(stockimage.Width, stockimage.Height);
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Size = new Size(image.Width + 16, image.Height + 39);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            Canvas.Refresh();
        }

        private void Canvas_Paint(object sender, PaintEventArgs e)
        {
            Canvas.CreateGraphics().DrawImage(image, new PointF(0, 0));
        }

        private void Canvas_LoadCompleted(object sender, AsyncCompletedEventArgs e)
        {
            Canvas.Refresh();
        }
    }

    public class DeformationMatrix
    {
        public List<Node> Nodes;
        private List<ColoredPolygon> Connections;

        public DeformationMatrix(Bitmap image)
        {
            Nodes = new List<Node>();
            int[,] NodeRefGrid = new int[image.Width+1, image.Height+1];
            for (int y = 0; y <= image.Height; y++)
            {
                for (int x = 0; x <= image.Width; x++)
                {
                    Node newNode = new Node(x - 0.5, y - 0.5);
                    NodeRefGrid[x, y] = Nodes.Count;
                    Nodes.Add(newNode);
                }
            }

            Connections = new List<ColoredPolygon>();
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int[] PolyNodeRefs = new int[4];
                    PolyNodeRefs[0] = NodeRefGrid[x, y];
                    PolyNodeRefs[1] = NodeRefGrid[x + 1, y];
                    PolyNodeRefs[2] = NodeRefGrid[x + 1, y + 1];
                    PolyNodeRefs[3] = NodeRefGrid[x, y + 1];
                    Connections.Add(new ColoredPolygon(PolyNodeRefs, image.GetPixel(x, y)));
                }
            }
        }

        public void ApplyTransformation(Func<Node, Node> transformation)
        {
            for (int i=0;i<Nodes.Count;i++)
            {
                Nodes[i] = transformation(Nodes[i]);
            }
        }

        public Bitmap ToBitmap(int width, int height)
        {
            Dictionary<Pixel, Color> ResolvedDataGrid = new Dictionary<Pixel, Color>();
            foreach (ColoredPolygon CP in Connections)
            {
                foreach (Pixel P in CP.GetAllOverlappingPixels(Nodes))
                {
                    ResolvedDataGrid[P] = CP.PolyColor;
                }
            }

            Bitmap Output = new Bitmap(width, height);
            for (int y=0;y<height;y++)
            {
                for (int x=0;x<width;x++)
                {
                    Output.SetPixel(x, y, Color.White);
                }
            }

            foreach (KeyValuePair<Pixel, Color> pair in ResolvedDataGrid)
            {
                Pixel p = pair.Key;
                if (p.X >= 0 && p.X < width && p.Y >= 0 && p.Y < height)
                    Output.SetPixel(p.X, p.Y, pair.Value);
            }

            return Output;
        }
    }



    public class Pixel
    {
        public int X;
        public int Y;

        public Pixel(int x, int y)
        {
            X = x;
            Y = y;
        }

        public override string ToString()
        {
            return string.Format("[{0}, {1}]", X, Y);
        }
    }

    public class ColoredPolygon
    {
        public int[] NodeRefs;
        public Color PolyColor;

        public ColoredPolygon(int[] refs, Color color)
        {
            NodeRefs = refs;
            PolyColor = color;
        }

        public Node[] FetchNodes(List<Node> NodeReference)
        {
            Node[] nodes = new Node[NodeRefs.Length];
            for (int i = 0; i < NodeRefs.Length; i++)
                nodes[i] = NodeReference[NodeRefs[i]];
            return nodes;
        }

        public bool ContainsPointOnFront(List<Node> NodeReference, double x, double y)
        {
            Node[] Nodes = FetchNodes(NodeReference);
            for (int i = 0; i < Nodes.Length; i++)
            {
                Node N1 = Nodes[i];
                Node N2 = Nodes[(i + 1) % Nodes.Length];
                double Polarity = (N2.X - N1.X) * (y - N1.Y) - (N2.Y - N1.Y) * (x - N1.X);
                if (Polarity < 0)
                    return false;
            }
            return true;
        }

        public bool ContainsPointOnBack(List<Node> NodeReference, double x, double y)
        {
            Node[] Nodes = FetchNodes(NodeReference);
            for (int i = Nodes.Length-1; i >= 0; i--)
            {
                Node N1 = Nodes[i];
                Node N2;
                if (i == 0)
                    N2 = Nodes[Nodes.Length - 1];
                else
                    N2 = Nodes[i - 1];
                double Polarity = (N2.X - N1.X) * (y - N1.Y) - (N2.Y - N1.Y) * (x - N1.X);
                if (Polarity < 0)
                    return false;
            }
            return true;
        }
        
        public bool ContainsPoint(List<Node> NodeReference, double x, double y)
        {
            return ContainsPointOnFront(NodeReference, x, y) || ContainsPointOnBack(NodeReference, x, y);
        }

        public Node GetMinimumPoint(List<Node> NodeReference)
        {
            Node[] Nodes = FetchNodes(NodeReference);
            Node Minimum = new Node(Nodes[0]);
            for (int i = 1;i<Nodes.Length;i++)
            {
                if (Nodes[i].X < Minimum.X)
                    Minimum.X = Nodes[i].X;
                if (Nodes[i].Y < Minimum.Y)
                    Minimum.Y = Nodes[i].Y;
            }
            return Minimum;
        }
        
        public Node GetMaximumPoint(List<Node> NodeReference)
        {
            Node[] Nodes = FetchNodes(NodeReference);
            Node Maximum = new Node(Nodes[0]);
            for (int i = 1; i < Nodes.Length; i++)
            {
                if (Nodes[i].X > Maximum.X)
                    Maximum.X = Nodes[i].X;
                if (Nodes[i].Y > Maximum.Y)
                    Maximum.Y = Nodes[i].Y;
            }
            return Maximum;
        }

        public Pixel GetRoundedMinimumPoint(List<Node> NodeReference)
        {
            Node MinNode = GetMinimumPoint(NodeReference);
            return new Pixel((int)Math.Ceiling(MinNode.X), (int)Math.Ceiling(MinNode.Y));
        }

        public Pixel GetRoundedMaximumPoint(List<Node> NodeReference)
        {
            Node MaxNode = GetMaximumPoint(NodeReference);
            return new Pixel((int)Math.Floor(MaxNode.X), (int)Math.Floor(MaxNode.Y));
        }

        public List<Pixel> GetAllOverlappingPixels(List<Node> NodeReference)
        {
            List<Pixel> Overlaps = new List<Pixel>();
            Pixel MinPixel = GetRoundedMinimumPoint(NodeReference);
            Pixel MaxPixel = GetRoundedMaximumPoint(NodeReference);

            for (int y = MinPixel.Y; y <= MaxPixel.Y; y++)
            {
                for (int x = MinPixel.X; x <= MaxPixel.X; x++)
                {
                    if (ContainsPoint(NodeReference, x, y))
                        Overlaps.Add(new Pixel(x, y));
                }
            }
            return Overlaps;
        }
    }

    public class Node
    {
        public double X;
        public double Y;

        public Node(double x, double y)
        {
            X = x;
            Y = y;
        }

        public Node(Node n)
        {
            X = n.X;
            Y = n.Y;
        }

        public override string ToString()
        {
            return string.Format("({0:0.00}, {1:0.00})", X, Y);
        }
    }
}

