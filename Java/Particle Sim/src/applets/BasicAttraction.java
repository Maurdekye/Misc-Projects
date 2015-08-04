package applets;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JApplet;

public class BasicAttraction extends JApplet{
	private static final long serialVersionUID = -6654953671558393230L;
	double[] vector = {266, 256, 1, 0};
	double[] orbit = {256, 256, -1};
	public void init() {
		setSize(1200,800);
	}
	
	double[] move(double[] vec, double[] orb) {
		double dir = Math.atan((vec[0]-orb[0])/(vec[1]-orb[1]))*(180/Math.PI);
		double xaffect = 0;
		double yaffect = 0;
		if (vec[0] >= orb[0]) {
			if (vec[1] < orb[1]) {
				yaffect = (dir/90)*orb[2];
				xaffect = ((90-dir)/90)*orb[2];
				System.out.println(dir+" => xAff: "+xaffect+" yAff: "+yaffect);
			} else {
				
			}
		} else {
			if (vec[1] <= orb[1]) {
				
			} else {
				
			}
		}
		vec[2] += xaffect;
		vec[3] += yaffect;
		vec[0] += vec[2];
		vec[1] += vec[3];
		return vec;
	}
	
	public void paint(Graphics screen) {
		screen.setColor(Color.green);
		for (int i=1;i<200;i++) {
			screen.drawRect((int) vector[0], (int) vector[1], 0, 0);
			screen.setColor(Color.RED);
			screen.drawRect((int) orbit[0]-1, (int) orbit[1]-1, 2, 2);
			vector = move(vector, orbit);
			screen.setColor(Color.black);
		}
	}
	
}
