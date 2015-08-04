import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

public class mandelbrot extends Applet implements MouseListener, MouseMotionListener{
	private static final long serialVersionUID = -2819735602198461003L;
	double juliaRe = -0.5;
	double juliaIm = 0.525;
	int iters = 1000;
	double multiplicative = Math.min(((double) getWidth()), ((double) getHeight())) / 4;
	public void init() {
		addMouseListener(this);
		addMouseMotionListener(this);
		setSize(1600, 800);
	}
	public int upX(double inval) {
		return (int) ((inval+2)*multiplicative);
	}
	public double downX(int inval) {
		return (double) (inval/multiplicative) - 2;
	}
	public int upY(double inval) {
		return (int) ((inval+2)*multiplicative);
	}
	public double downY(int inval) {
		return  (double) (inval/multiplicative) - 2;
	}
	
	public int mandel(double real, double imag, double jReal, double jImag, int iters) {
		int to_sender = 0;
		for (int i=0;i<iters;i++) {
			if ((real*real) + (imag*imag) > 4)
				break;
			double backReal = real;
			double backImag = imag;
			real = ( (backReal*backReal) - (backImag*backImag) ) + jReal;
			imag = ( 2 * backImag * backReal ) + jImag;
			to_sender = i+1;
		}
		return to_sender;
	}
	
	public void paint(Graphics s) {
		float diffFactor = 40.0F;
		for(int re=0;re<((double) getHeight());re++) {
			for(int im=0;im<((double) getHeight());im++) {
				int m = mandel(downX(re), downY(im), downX(re), downY(im), iters);
				if (m < iters) {
					s.setColor(Color.getHSBColor(1, 1, m/diffFactor));
					s.drawLine(re, im, re, im);
				}
			}
		}
		for(int re=getHeight();re<getWidth();re++) {
			for(int im=0;im<((double) getHeight());im++) {
				int m = mandel(downX(re)-4, downY(im), juliaRe, juliaIm, iters);
				if (m < iters) {
					s.setColor(Color.getHSBColor(1, 1, m/diffFactor));
					s.drawLine(re, im, re, im);
				}
			}
		}
	}
	
	public double biDeciRound(double inval) {
		inval = 100 * inval;
		inval = Math.round(inval);
		return inval/100.0;
	}
	
	public void mouseDragged(MouseEvent e) {}
	public void mouseMoved(MouseEvent e) {}
	public void mouseClicked(MouseEvent mouse) {
		if (mouse.getX() < ((double) getHeight())) {
			juliaRe = downX(mouse.getX());
			juliaIm = downY(mouse.getY());

			String s1 = "";
			String s2 = "";
			if (juliaRe >= 0)
				s1 = "+";
			if (juliaIm >= 0)
				s2 = "+";
			System.out.println(s1 + Double.toString(biDeciRound(juliaRe)) + s2 + Double.toString(biDeciRound(juliaIm))+"i");
			repaint();
		}
	}
	public void mouseEntered(MouseEvent arg0) {}
	public void mouseExited(MouseEvent arg0) {}
	public void mousePressed(MouseEvent arg0) {}
	public void mouseReleased(MouseEvent arg0) {}
	Thread clock = new Thread() { public void run() { while (true) {
		
	try {Thread.sleep(15);}
	catch (InterruptedException e) {e.printStackTrace();}
	}}};
}
