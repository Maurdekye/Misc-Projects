import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class GameOfLife extends Applet implements MouseListener, MouseMotionListener, KeyListener{

    public Set<Vect> cells = new HashSet<>();
    public Vect gridsize = new Vect(32, 32);
    public Set<Vect> surroundModifs = new HashSet<>();
    public Vect sizeMult;

    public boolean paused = false;
    public boolean step = false;

    public Vect lastCheck = new Vect(0, 0);

    // Inherited Methods

    public void init() {

        setSize(512, 512);
        addMouseListener(this);
        addMouseMotionListener(this);
        addKeyListener(this);
        setFocusable(true);

        for (int x=-1;x<=1;x++) {
            for (int y=-1;y<=1;y++) {
                if (x == 0 && y == 0) continue;
                surroundModifs.add(new Vect(x, y));
            }
        }
        sizeMult = new Vect(getWidth() / gridsize.x, getHeight() / gridsize.y);

        new Thread() {
            public synchronized void run () {
                while (true) {
                    try {
                        if (!paused || step) {
                            step = false;

                            // Compile a list of squares to check in

                            Set<Vect> toCheck = new HashSet<>();
                            for (Vect cell : cells) {
                                for (Vect toMod : surroundModifs) {
                                    Vect toAdd = cell.add(toMod);
                                    if (!toAdd.isContained(toCheck))
                                        toCheck.add(toAdd);
                                }
                            }

                            // Check each square to see if it lives or dies

                            Set<Vect> newCells = new HashSet<>();
                            for (Vect cell : toCheck) {
                                boolean isAlive = cell.isContained(cells);

                                int neighbors = 0;
                                for (Vect toMod : surroundModifs) {
                                    if (cell.add(toMod).isContained(cells))
                                        neighbors++;
                                }

                                if (neighbors == 3 || (neighbors == 2 && isAlive))
                                    newCells.add(cell.clone());
                            }

                            cells.clear();
                            cells.addAll(newCells);
                        }
                        repaint();
                        Thread.sleep(100);
                    } catch(InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();

    }

    public void paint(Graphics g) {
        g.setColor(new Color(150, 150, 150));
        g.fillRect(0, 0, getWidth(), getHeight());

        for (int x = 0; x < gridsize.x; x++) {
            for (int y = 0; y < gridsize.y; y++) {
                Vect lookAt = new Vect(x, y);
                Vect paintAt = lookAt.multiply(sizeMult);

                if (lookAt.isContained(cells)) {
                    g.setColor(Color.YELLOW);
                    g.fillRect((int) (paintAt.x + sizeMult.x/8), (int) (paintAt.y + sizeMult.y/8),
                            (int) (sizeMult.x * 3/5), (int) (sizeMult.y * 3/5));
                } else {
                    g.setColor(new Color(120, 120, 120));
                    g.fillRect((int) (paintAt.x + sizeMult.x/10), (int) (paintAt.y + sizeMult.y/10),
                            (int) (sizeMult.x * 4/5), (int) (sizeMult.y * 4/5));
                }
            }
        }
    }

    @Override
    public void mousePressed(MouseEvent e) {
        Vect clickVec = normalize(new Vect(e.getX(), e.getY()));
        if (e.getButton() == 1) {
            if (!clickVec.isContained(cells)) {
                cells.add(clickVec);
            }
        } else if (e.getButton() == 3) {
            if (clickVec.isContained(cells))
                cells.remove(clickVec);
        }
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        Vect clickVec = normalize(new Vect(e.getX(), e.getY()));
        if (e.getButton() == 1) {
            if (!clickVec.isContained(cells))
                cells.add(clickVec);
        } else if (e.getButton() == 3) {
            if (clickVec.isContained(cells))
                cells.remove(clickVec);
        }
    }

    @Override
    public void keyPressed(KeyEvent e) {
        int code = e.getKeyCode();
        System.out.println(code);
        if (code == 32)
            paused = !paused;
        else if (code  == 39)
            step = true;
        else if (code == 67)
            cells.clear();
    }

    // Private Methods

    public Vect normalize(Vect inv) {
        Vect ret = new Vect(inv.x / sizeMult.x, inv.y / sizeMult.y);
        ret.x = Math.floor(ret.x);
        ret.y = Math.floor(ret.y);
        return ret;
    }

    // Unused Methods

    public void mouseClicked(MouseEvent e) {}
    public void mouseReleased(MouseEvent e) {}
    public void mouseEntered(MouseEvent e) {}
    public void mouseExited(MouseEvent e) {}
    public void mouseMoved(MouseEvent e) {}
    public void keyTyped(KeyEvent e) {}
    public void keyReleased(KeyEvent e) {}
}
