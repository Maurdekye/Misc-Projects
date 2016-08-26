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
        public static int ParticleCount = 10;

        private Thread RenderThread;
        private Thread SimulationThread;

        private List<Particle> Particles;
        private Bounds EnvironmentBounds;

        private FramerateMeter FrameCounter;

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
            RenderThread = new Thread(new ThreadStart(Render));
            SimulationThread = new Thread(new ThreadStart(Simulate));
            Paused = false;
            FrameAdvance = false;
            FrameCounter = new FramerateMeter();
            Size CSize = Canvas.Bounds.Size;
            EnvironmentBounds = new Bounds(new PointF(0, 0), new PointF(CSize.Width, CSize.Height));
            Particles = new List<Particle>();
            for (int i = 0; i < ParticleCount; i++)
                Particles.Add(new Particle(10 + i*4, RandomPointFInBounds(EnvironmentBounds), RandomInUnitCircle().Times(1000)));
        }

        public void InitializeThreads()
        {
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

            // Clear Background
            g.FillRectangle(Brushes.White, Canvas.Bounds);

            // Draw particles
            foreach (Particle P in Particles)
                g.FillEllipse(Brushes.Blue, new Rectangle(P.Position.Minus(P.Radius).ToPoint(), new Size((int)P.Radius*2, (int)P.Radius*2)));

            // Update and Draw Framerate
            g.DrawString(FrameCounter.TickFramerate().ToString(), new Font("Helvetica", 10), Brushes.Black, new PointF(10, 10));

            // Draw Pause Icon
            if (Paused)
                g.DrawString("=", new Font("Helvetica", 80, FontStyle.Bold), Brushes.Black, new PointF(190, 170));
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
        public PointF Velocity { get; private set; } = new PointF(0, 0);
        public PointF Position { get; private set; } = new PointF(0, 0);

        private List<Particle> CollidedRecently = new List<Particle>();

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
            PreVelocity = PreVelocity.Plus(new PointF(0, 1000f * DeltaTime));
        }

        private void TestForWallCollision(float DeltaTime, Bounds EnvironmentBounds)
        {
            if (PrePosition.X < EnvironmentBounds.Lower.X + Radius
                || PrePosition.X > EnvironmentBounds.Upper.X - Radius)
            {
                PreVelocity.X = -PreVelocity.X;
            }
            if (PrePosition.Y < EnvironmentBounds.Lower.Y + Radius
                || PrePosition.Y > EnvironmentBounds.Upper.Y - Radius)
            {
                PreVelocity.Y = -PreVelocity.Y;
            }
        }

        private void TestForParticleCollision(float DeltaTime, List<Particle> Particles)
        {
            foreach (Particle Other in Particles)
            {
                if (Other == this)
                    continue;
                bool HasRecentlyCollided = CollidedRecently.Contains(Other);
                if (Other.Position.Distance(Position) <= Other.Radius + this.Radius)
                {
                    if (HasRecentlyCollided)
                        continue;
                    else
                    {
                        SimulateCollision(DeltaTime, Other);
                        CollidedRecently.Add(Other);
                    }
                }
                else if (HasRecentlyCollided)
                {
                    CollidedRecently.Remove(Other);
                }

            }
        }

        private void SimulateCollision(float DeltaTime, Particle Other)
        {
            float M1 = Mass(); // Mass 1
            float M2 = Other.Mass(); // Mass 2
            float S1 = Velocity.Magnitude(); // Speed 1
            float S2 = Other.Velocity.Magnitude(); // Speed 2
            float A1 = Velocity.Angle(); // Angle 1
            float A2 = Other.Velocity.Angle(); // Angle 2
            PointF DeltaVec = Other.Position.Minus(Position);
            float CA = DeltaVec.Angle(); // Collision Angle
            float DA1 = A1 - CA; // Delta Angle 1
            float DA2 = A2 - CA; // Delta Angle 2

            double FractionalCalculation = (S1 * Math.Cos(DA1) * (M1 - M2) + 2 * M2 * S2 * Math.Cos(DA2)) / (M1 + M2);
            PreVelocity = new PointF(
                (float)(FractionalCalculation * Math.Cos(CA) + S1 * Math.Sin(DA1) * Math.Cos(CA + Math.PI / 2)),
                (float)(FractionalCalculation * Math.Sin(CA) + S1 * Math.Sin(DA1) * Math.Sin(CA + Math.PI / 2))
                );
        }

        private void RemoveFromOutOfBounds(Bounds EnvironmentBounds)
        {
            if (PrePosition.X < EnvironmentBounds.Lower.X + Radius)
                PrePosition.X = EnvironmentBounds.Lower.X + Radius + 1;
            if (PrePosition.X > EnvironmentBounds.Upper.X - Radius)
                PrePosition.X = EnvironmentBounds.Upper.X - Radius - 1;
            if (PrePosition.Y < EnvironmentBounds.Lower.Y + Radius)
                PrePosition.Y = EnvironmentBounds.Lower.Y + Radius + 1;
            if (PrePosition.Y > EnvironmentBounds.Upper.Y - Radius)
                PrePosition.Y = EnvironmentBounds.Upper.Y - Radius - 1;
        }

        private void RemoveFromOtherParticles(List<Particle> Particles)
        {
            foreach (Particle Other in Particles)
            {
                if (Other == this)
                    continue;
                if (CollidedRecently.Contains(Other) && Other.Position.Distance(Position) <= Other.Radius + this.Radius)
                {
                    float Overlap = (Other.Radius + this.Radius) - Other.Position.Distance(Position);
                    PointF DirectionVec = Position.Minus(Other.Position).Normalized().Times(Overlap);
                    PrePosition = PrePosition.Plus(DirectionVec.Times(0.5f));
                }
            }
        }

        public void SimulateNextFrame(float DeltaTime, Bounds EnvironmentBounds, List<Particle> Particles)
        {
            PreVelocity = Velocity;
            PrePosition = SimulateLinearAcceleration(DeltaTime);
            TestForWallCollision(DeltaTime, EnvironmentBounds);
            TestForParticleCollision(DeltaTime, Particles);
            SimulateGravity(DeltaTime);
            PrePosition = SimulateLinearAcceleration(DeltaTime);
            RemoveFromOtherParticles(Particles);
            RemoveFromOutOfBounds(EnvironmentBounds);
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

    public class FramerateMeter : Stopwatch
    {
        private long LastTick = 0;

        public FramerateMeter() : base() { }

        public float TickFramerate(int decimals=2)
        {
            if (!IsRunning)
                Start();
            long TickDelta = ElapsedTicks - LastTick;
            double SecondDelta = TickDelta / (double)Frequency;
            double RawFramerate = 1.0 / SecondDelta;
            float FinalFramerate = (float)Math.Round(RawFramerate, decimals);
            LastTick = ElapsedTicks;
            return FinalFramerate;
        }
    }

    public static class PointFAlgebra
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

        public static float Angle(this PointF P)
        {
            return (float)Math.Atan2(P.Y, P.X);
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

        public static PointF Perpendicular(this PointF P)
        {
            return new PointF(-P.Y, P.X);
        }

        public static Point ToPoint(this PointF P)
        {
            return new Point((int)P.X, (int)P.Y);
        }
    }
}
