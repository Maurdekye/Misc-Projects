namespace SoftwareRenderer
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.RenderTarget = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.RenderTarget)).BeginInit();
            this.SuspendLayout();
            // 
            // RenderTarget
            // 
            this.RenderTarget.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RenderTarget.Location = new System.Drawing.Point(0, 0);
            this.RenderTarget.Name = "RenderTarget";
            this.RenderTarget.Size = new System.Drawing.Size(795, 320);
            this.RenderTarget.TabIndex = 0;
            this.RenderTarget.TabStop = false;
            this.RenderTarget.UseWaitCursor = true;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(795, 320);
            this.Controls.Add(this.RenderTarget);
            this.Name = "MainForm";
            this.Text = "Software Rendering Test";
            this.UseWaitCursor = true;
            ((System.ComponentModel.ISupportInitialize)(this.RenderTarget)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox RenderTarget;
    }
}

