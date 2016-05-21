using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace BeatTracker
{
    public partial class window : Form
    {
        public long startTime = -1;
        public List<float> beats = new List<float>();

        public window()
        {
            InitializeComponent();
        }

        private void window_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (e.KeyChar == ' ')
            {
                float time = curTime();
                beats.Add(time);
                int count = beats.Count ;
                float bpm = count / (time / 60);
                Graphics g = canvas.CreateGraphics();
                g.Clear(Color.White);
                g.DrawString(bpm + "", new Font("Arial", 12 , FontStyle.Regular), new SolidBrush(Color.Black), new PointF(10, 10));
                g.DrawString(time + "", new Font("Arial", 12, FontStyle.Regular), new SolidBrush(Color.Black), new PointF(10, 30));
                int h = canvas.Height / 2;
                int w = canvas.Width / count;
                float spb = time / count;
                g.DrawRectangle(Pens.Black, new Rectangle(0, h, canvas.Width, 1));
                int x = 0;
                float y = 0;
                foreach (float beat in beats)
                {
                    int bar = (int) ((y - beat) * h);
                    if (bar < 0)
                        g.FillRectangle(new SolidBrush(Color.Gray), new Rectangle(x, h + bar, w, -bar));
                    else
                        g.FillRectangle(new SolidBrush(Color.Gray), new Rectangle(x, h, w, bar));
                    x += w;
                    y += spb;
                }
            }
        } 

        public float curTime()
        {
            if (startTime < 0)
                startTime = DateTime.Now.Ticks;
            return (DateTime.Now.Ticks - startTime) / (float)TimeSpan.TicksPerSecond;
        }
    }
}
