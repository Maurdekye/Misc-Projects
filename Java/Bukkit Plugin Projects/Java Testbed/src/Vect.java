public class Vect {
    double x;
    double y;
    public Vect(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Vect clone() {
        return new Vect(this.x, this.y);
    }

    public String toString() {
        return "(" + x + ", " + y + ")";
    }

    public boolean equals(Object other) {
        if (!(other instanceof Vect)) return false;
        Vect otherVect = (Vect) other;
        return otherVect.x == this.x && otherVect.y == this.y;
    }

    public Vect add(Vect other) {
        return new Vect(this.x + other.x, this.y + other.y);
    } public Vect add(double num) {return add(new Vect(num, num));}
    public Vect subtract(Vect other) {
        return new Vect(this.x - other.x, this.y - other.y);
    } public Vect subtract(double num) {return subtract(new Vect(num, num));}
    public Vect multiply(Vect other) {
        return new Vect(this.x * other.x, this.y * other.y);
    } public Vect multiply(double num) {return multiply(new Vect(num, num));}
    public Vect divide(Vect other) {
        return new Vect(this.x / other.x, this.y / other.y);
    } public Vect divide(double num) {return divide(new Vect(num, num));}

    public boolean isIn(Vect other) {
        if (this.y < 0) return false;
        if (this.x < 0) return false;
        if (this.y >= other.y) return false;
        if (this.x >= other.x) return false;
        return true;
    }

    public boolean isContained(Iterable<Vect> checkIn) {
        for (Vect other : checkIn)
            if (other.equals(this)) return true;
        return false;
    }
}
