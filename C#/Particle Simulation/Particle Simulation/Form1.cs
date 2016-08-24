using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static Particle_Simulation.Util;

namespace Particle_Simulation
{
    public partial class Window : Form
    {
        public static float RenderRate = 1f / 60;
        public static float SimulationRate = 1f / 120;
        public static float Timescale = 1f;
        public static int ParticleCount = 2;

        private Thread RenderThread;
        private Thread SimulationThread;

        private List<Particle> Particles;
        private Bounds EnvironmentBounds;

        private Stopwatch FramerateCounter;
        private long LastFrameTime;

        private bool Paused;
        private bool FrameAdvance;

        public Window()
        {
            InitializeComponent();
            CreateEnvironment();
        }

        private void Window_Activated(object sender, EventArgs e)
        {
            InitializeThreads();
        }

        private void Window_Deactivate(object sender, EventArgs e)
        {
            DestroyThreads();
        }

        public void CreateEnvironment()
        {
            Paused = false;
            FrameAdvance = false;
            FramerateCounter = new Stopwatch();
            FramerateCounter.Start();
            LastFrameTime = FramerateCounter.ElapsedTicks;
            Size CSize = Canvas.Bounds.Size;
            EnvironmentBounds = new Bounds(new PointF(0, 0), new PointF(CSize.Width, CSize.Height));
            Particles = new List<Particle>();
            /*for (int i = 0; i < ParticleCount; i++)
            {
                Particles.Add(new Particle(RandomFloat()*30 + 10, RandomPointFInBounds(EnvironmentBounds), RandomInUnitCircle().Times(1000)));
            }*/
            Particles.Add(new Particle(15, new PointF(150, 250), new PointF(300, 0)));
            Particles.Add(new Particle(15, new PointF(250, 270), new PointF(0, 0)));
        }

        public void InitializeThreads()
        {
            RenderThread = new Thread(new ThreadStart(Render));
            SimulationThread = new Thread(new ThreadStart(Simulate));
            RenderThread.Start();
            SimulationThread.Start();
        }

        public void DestroyThreads()
        {
            RenderThread.Abort();
            SimulationThread.Abort();
        }

        public void Render()
        {
            while(true)
            {
                Thread.Sleep((int)(RenderRate * 1000));
                Repaint();
            }
        }

        public void Simulate()
        {
            while(true)
            {
                Thread.Sleep((int)(SimulationRate * 1000 * (1f / Timescale)));
                if (Paused && !FrameAdvance)
                    continue;
                if (FrameAdvance)
                    FrameAdvance = false;
                List<Thread> ParticleThreads = new List<Thread>();
                foreach (Particle P in Particles)
                {
                    Thread CurThread = new Thread(new ThreadStart(delegate
                    {
                        P.SimulateNextFrame(SimulationRate, EnvironmentBounds, Particles);
                    }));
                    ParticleThreads.Add(CurThread);
                    CurThread.Start();
                }
                bool AllDone;
                do
                {
                    AllDone = true;
                    foreach (Thread T in ParticleThreads)
                    {
                        if (T.IsAlive)
                        {
                            AllDone = false;
                            break;
                        }
                    }
                } while (!AllDone);
                foreach (Particle P in Particles)
                {
                    P.ApplySimulation();
                }
            }
        }

        private void Repaint()
        {
            if (!Canvas.IsDisposed)
                Canvas.Invoke(new Action(() => Canvas.Refresh()));
        }

        private void Canvas_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            // Refresh Background
            g.FillRectangle(Brushes.White, Canvas.Bounds);
            // Render particles
            foreach (Particle P in Particles)
            {
                Brush PColor = Brushes.Blue;
                if (P.Colliding)
                    PColor = Brushes.Red;
                g.FillEllipse(PColor, new Rectangle(P.Position.Minus(P.Radius).ToPoint(), new Size((int)P.Radius*2, (int)P.Radius*2)));
            }
            // Calculate and draw framerate
            long CurrentFrameTime = FramerateCounter.ElapsedTicks;
            long FrameDelta = CurrentFrameTime - LastFrameTime;
            double DeltaSeconds = FrameDelta / (double)Stopwatch.Frequency;
            double InstantaneousFramerate = Math.Round(1 / DeltaSeconds, 2);
            g.DrawString(InstantaneousFramerate.ToString(), new Font("Helvetica", 10), Brushes.Black, new PointF(10, 10));
            LastFrameTime = CurrentFrameTime;
            // Draw Paused Icon
            if (Paused)
            {
                g.DrawString("=", new Font("Helvetica", 80, FontStyle.Bold), Brushes.Black, new PointF(170, 170));
            }
        }

        private void Window_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == ' ')
                Paused = !Paused;
            else if (e.KeyChar == 'f')
                FrameAdvance = true;

        }
    }

    public class Particle
    {

        public float Radius = 1;

        private PointF PreVelocity = new PointF(0, 0);
        private PointF PrePosition = new PointF(0, 0);
        public PointF Velocity = new PointF(0, 0);
        public PointF Position = new PointF(0, 0);

        public bool Colliding { get; private set; }

        public Particle() { }

        public Particle(float Radius)
        {
            this.Radius = Radius;
        }

        public Particle(float Radius, PointF Position)
        {
            this.Radius = Radius;
            this.Position = Position;
        }

        public Particle(float Radius, PointF Position, PointF Velocity)
        {
            this.Radius = Radius;
            this.Position = Position;
            this.Velocity = Velocity;
        }
        
        public float Mass()
        {
            return Radius * Radius * (float)Math.PI;
        }

        private PointF SimulateLinearAcceleration(float DeltaTime)
        {
            return Position.Plus(Velocity.Times(DeltaTime));
        }

        private void SimulateGravity(float DeltaTime)
        {
            PreVelocity.Minus(new PointF(0, 981f * DeltaTime));
        }

        private void SimulateCollision(float DeltaTime, Particle Other)
        {
            float MyMass = Mass();
            float OtherMass = Other.Mass();
            PreVelocity = new PointF(
                (Velocity.X * (MyMass - OtherMass) + (2 * OtherMass * Other.Velocity.X)) / (MyMass + OtherMass),
                (Velocity.Y * (MyMass - OtherMass) + (2 * OtherMass * Other.Velocity.Y)) / (MyMass + OtherMass));
        }

        public void SimulateNextFrame(float DeltaTime, Bounds EnvironmentBounds, List<Particle> Particles)
        {
            PrePosition = Position;
            PreVelocity = Velocity;
            PointF NewPos = SimulateLinearAcceleration(DeltaTime);
            bool Altered = false;
            Colliding = false;

            // Wall Collisions
            if (NewPos.X < EnvironmentBounds.Lower.X + Radius
                || NewPos.X > EnvironmentBounds.Upper.X - Radius)
            {
                PreVelocity.X = -PreVelocity.X;
                Altered = true;
            }
            if (NewPos.Y < EnvironmentBounds.Lower.Y + Radius
                || NewPos.Y > EnvironmentBounds.Upper.Y - Radius)
            {
                PreVelocity.Y = -PreVelocity.Y;
                Altered = true;
            }

            // Particle Collisions
            foreach (Particle Other in Particles)
            {
                if (Other == this)
                    continue;
                if (Other.Position.Distance(Position) < Other.Radius + this.Radius)
                {
                    SimulateCollision(DeltaTime, Other);
                    Altered = true;
                    Colliding = true;
                }
            }

            if (Altered)
                NewPos = SimulateLinearAcceleration(DeltaTime);
            PrePosition = NewPos;

            // Remove Balls From Walls
            if (PrePosition.X < EnvironmentBounds.Lower.X + Radius)
                PrePosition.X = EnvironmentBounds.Lower.X + Radius + 1;
            if (PrePosition.X > EnvironmentBounds.Upper.X - Radius)
                PrePosition.X = EnvironmentBounds.Upper.X - Radius - 1;
            if (PrePosition.Y < EnvironmentBounds.Lower.Y + Radius)
                PrePosition.Y = EnvironmentBounds.Lower.Y + Radius + 1;
            if (PrePosition.Y > EnvironmentBounds.Upper.Y - Radius)
                PrePosition.Y = EnvironmentBounds.Upper.Y - Radius - 1;
        }

        public void ApplySimulation()
        {
            Position = PrePosition;
            Velocity = PreVelocity;
        }
        
    }

    public class Bounds
    {
        public PointF Lower { get; private set; }
        public PointF Upper { get; private set; }
        public PointF Range { get; private set; }

        public Bounds()
        {
            Init(new PointF(0, 0), new PointF(0, 0));
        }

        public Bounds(PointF Upper)
        {
            Init(new PointF(0, 0), Upper);
        }

        public Bounds(PointF Lower, PointF Upper)
        {
            Init(Lower, Upper);
        }

        private void Init(PointF Lower, PointF Upper)
        {
            this.Lower = new PointF(Math.Min(Lower.X, Upper.X), Math.Min(Lower.Y, Upper.Y));
            this.Upper = new PointF(Math.Max(Lower.X, Upper.X), Math.Max(Lower.Y, Upper.Y));
            Range = new PointF(Upper.X - Lower.X, Upper.Y - Lower.Y);
        }

        public static Bounds operator +(Bounds B, PointF P)
        {
            return new Bounds(B.Lower.Plus(P), B.Upper.Plus(P));
        }

        public static Bounds operator *(Bounds B, PointF P)
        {
            return new Bounds(B.Lower.Times(P), B.Upper.Times(P));
        }

        public bool Contains(PointF P)
        {
            if (Lower.X > P.X) return false;
            if (Lower.Y > P.Y) return false;
            if (Upper.X < P.X) return false;
            if (Upper.Y < P.Y) return false;
            return true;
        }
            
    }

    public static class Util
    {
        private static Random RNG = new Random();

        public static float RandomFloat()
        {
            return (float)RNG.NextDouble();
        }

        public static PointF RandomPointF()
        {
            return new PointF(RandomFloat(), RandomFloat());
        }

        public static PointF RandomPointFInBounds(Bounds Bound)
        {
            return RandomPointF().Times(Bound.Range).Plus(Bound.Lower);
        }

        public static PointF RandomInUnitCircle()
        {
            double Angle = RandomFloat() * Math.PI * 2f;
            double Distance = Math.Sqrt(RandomFloat());
            PointF AtPerimiter = new PointF((float)Math.Sin(Angle), (float)Math.Cos(Angle));
            return AtPerimiter.Times((float)Distance);
        }
    }

    public static class Extensions
    {
        public static PointF Plus(this PointF A, PointF B)
        {
            return new PointF(A.X + B.X, A.Y + B.Y);
        }

        public static PointF Plus(this PointF P, float S)
        {
            return new PointF(P.X + S, P.Y + S);
        }

        public static PointF Minus(this PointF A, PointF B)
        {
            return new PointF(A.X - B.X, A.Y - B.Y);
        }

        public static PointF Minus(this PointF P, float S)
        {
            return new PointF(P.X - S, P.Y - S);
        }

        public static PointF AbsoluteValue(this PointF P)
        {
            return new PointF(Math.Abs(P.X), Math.Abs(P.Y));
        }

        public static PointF Times(this PointF A, PointF B)
        {
            return new PointF(A.X * B.X, A.Y * B.Y);
        }

        public static PointF Times(this PointF P, float S)
        {
            return new PointF(P.X * S, P.Y * S);
        }

        public static float Distance(this PointF A, PointF B)
        {
            PointF Delta = A.Minus(B).AbsoluteValue();
            return (float)Math.Sqrt(Delta.X * Delta.X + Delta.Y * Delta.Y);
        }

        public static float Magnitude(this PointF A)
        {
            return A.Distance(new PointF(0, 0));
        }

        public static PointF Normalized(this PointF A)
        {
            float Mag = A.Magnitude();
            if (Mag == 0)
                return new PointF(0, 0);
            else
                return A.Times(1f / Mag);
        }

        public static float Dot(this PointF A, PointF B)
        {
            return A.X * B.X + A.Y * B.Y;
        }

        public static Point ToPoint(this PointF P)
        {
            return new Point((int)P.X, (int)P.Y);
        }
    }
}
