import java.applet.Applet;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.Random;

public class FlappyBird extends Applet implements KeyListener {

    Vect pos;
    boolean gameover = false;
    double upMomentum = 0;
    int clock = 0;

    ArrayList<Pipe> pipes = new ArrayList<>();

    public void init() {
        setSize(288, 512);
        addKeyListener(this);
        setFocusable(true);

        pos = new Vect(0, getHeight()/2);

        new Thread() {
            public void run() {
                while(true) {
                    try {

                        clock++;
                        System.out.println(clock + " " + (clock%150));
                        if (clock%150 == 0) {
                            pipes.add(new Pipe(getWidth(), getHeight()));
                            System.out.println(pipes.size());
                        }

                        upMomentum -= 0.2;
                        pos = pos.add(new Vect(1, upMomentum));

                        for (Pipe p : cloneArrayList(pipes)) {
                            p.progress--;
                            if (p.progress <= 0)
                                pipes.remove(p);
                            else if (p.progress < getWidth()/4 + 50 && p.progress > getWidth()/4) {
                                if (pos.y < p.height || pos.y > p.height + 80)
                                    gameover = true;
                            }
                        }

                        if (pos.y > getHeight()) gameover = true;

                        repaint();
                        if (gameover) break;
                        Thread.sleep(16);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();
    }

    public void paint(Graphics g) {
        g.setColor(Color.BLACK);
        if (gameover) g.drawString("Game Over!", getWidth()/2, getHeight()/2);
        g.setColor(Color.CYAN);
        g.fillRect(0, 0, getWidth(), getHeight());
        g.setColor(Color.YELLOW);
        g.fillOval(getWidth()/4, getHeight() - ((int) pos.y), 50, 35);
        g.setColor(Color.GREEN);
        for (Pipe p : pipes) {
            g.fillRect(p.progress, getHeight(), 50, getHeight()/2 - 50);
            g.fillRect(p.progress, getHeight()/2, 50, getHeight()/2);
        }
    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (gameover) return;
        int code = e.getKeyCode();
        if (code == 32 || code == 38 || code == 87) {
            upMomentum = 4;
        }
    }

    public <T> ArrayList<T> cloneArrayList(ArrayList<T> toClone) {
        ArrayList<T> toGive = new ArrayList<>();
        for (T item : toClone)
            toGive.add(item);
        return toGive;
    }

    public void keyReleased(KeyEvent e) {}
    public void keyTyped(KeyEvent e) {}
}

class Pipe {
    int progress;
    int height;

    public Pipe(int width, int height) {
        this.progress = width + 100;
        this.height = Math.abs(new Random().nextInt()) % height;
    }
}
