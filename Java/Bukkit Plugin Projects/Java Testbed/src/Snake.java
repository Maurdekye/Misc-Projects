import java.applet.Applet;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.*;

public class Snake extends Applet implements KeyListener {

    Vect size = new Vect(32, 32);
    Vect sizemult;
    boolean gameover = false;
    boolean paused = false;

    ArrayList<Vect> moveQueue = new ArrayList<>();

    Random rand = new Random();

    Vect headpos = size.divide(2);
    Vect treatpos = new Vect(Math.abs(rand.nextInt()) % size.x, Math.abs(rand.nextInt()) % size.y);
    Vect movedir = new Vect(0, -1);
    int length = 2;
    HashMap<Vect, Integer> tail = new HashMap<>();

    public void init() {
        setSize(512, 512);
        addKeyListener(this);
        setFocusable(true);
        sizemult = new Vect(getWidth() / size.x, getHeight() / size.y);


        new Thread() {
            public void run() {
                while (true) {
                    try {
                        if (paused || gameover) {
                            Thread.sleep(50);
                            repaint();
                            continue;
                        }

                        if (moveQueue.size() > 0) {
                            movedir = moveQueue.get(0);
                            moveQueue.remove(0);
                        }

                        tail.put(headpos.clone(), length);
                        headpos = headpos.add(movedir);
                        for (Vect key : cloneSet(tail.keySet())) {
                            int val = tail.get(key);
                            tail.remove(key);
                            if (val > 0)
                                tail.put(key, val-1);
                        }

                        if (headpos.equals(treatpos)) {
                            length++;
                            treatpos = tail.keySet().iterator().next().clone();
                            while (treatpos.isContained(tail.keySet()))
                                treatpos = new Vect(Math.abs(rand.nextInt()) % size.x, Math.abs(rand.nextInt()) % size.y);
                        }

                        // Failure Checks
                        if (headpos.isContained(tail.keySet())) {
                            gameover = true;
                        } else if (!headpos.isIn(size)) {
                            gameover = true;
                        }

                        repaint();

                        if (gameover) break;
                        Thread.sleep(200);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();
    }

    public void paint(Graphics g) {
        g.setColor(Color.BLACK);
        if (gameover)
            g.drawString("Game Over!", getWidth()/2, getHeight()/2);
        else if (paused)
            g.drawString("Paused", getWidth()/2, getHeight()/2);
        for (Vect v : tail.keySet())
            g.fillRect((int) (v.x * sizemult.x), (int) (v.y * sizemult.y), (int) sizemult.x, (int) sizemult.y);
        g.setColor(Color.GRAY);
        g.fillRect((int) (headpos.x * sizemult.x), (int) (headpos.y * sizemult.y), (int) sizemult.x, (int) sizemult.y);
        g.setColor(Color.RED);
        g.fillRect((int) (treatpos.x * sizemult.x), (int) (treatpos.y * sizemult.y), (int) sizemult.x, (int) sizemult.y);
    }

    public <T> Set<T> cloneSet(Set<T> toClone) {
        HashSet<T> toGive = new HashSet<>();
        for (T item : toClone)
            toGive.add(item);
        return toGive;
    }

    @Override
    public void keyPressed(KeyEvent e) {
        int code = e.getKeyCode();
        System.out.println(code);
        if (code == 32) {
            paused = !paused;
        } else if (code == 82) { // reset
            gameover = false;
            paused = false;
            tail.clear();
            headpos = size.divide(2);
            treatpos = new Vect(Math.abs(rand.nextInt()) % size.x, Math.abs(rand.nextInt()) % size.y);
            movedir = new Vect(0, -1);
            length = 2;
        }
        if (!paused) {
            if (code == 38 || code == 87) { // up
                moveQueue.add(new Vect(0, -1));
            } else if (code == 40 || code == 83) { // down
                moveQueue.add(new Vect(0, 1));
            } else if (code == 37 || code == 65) { // left
                moveQueue.add(new Vect(-1, 0));
            } else if (code == 68 || code == 39) { // right
                moveQueue.add(new Vect(1, 0));
            }
        }
    }

    public void keyReleased(KeyEvent e) {}
    public void keyTyped(KeyEvent e) {}
}

