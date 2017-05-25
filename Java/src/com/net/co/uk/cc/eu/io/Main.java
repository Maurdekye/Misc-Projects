package com.net.co.uk.cc.eu.io;

import java.io.IOException;
import java.util.Optional;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {

        int maxnum = 100;
        int maxnumlen = (int) Math.log10(maxnum * maxnum) + 1;
        for (int y = 1; y <= maxnum; y++) {
            for (int x = 1; x <= maxnum; x++) {
                int curnumlen = (int) Math.log10(x * y) + 1;
                int maxcolumnnumlen = (int) Math.log10(x * maxnum) + 1;
                String format = "";
                // we want all final numbers to have at least a length of maxmaxnum
                // we want to put enough spaces in front to make the final length of all numbers the same
                // we will take the difference between curnumlen and maxnumlen to determine how many spaces to put
                for (int s = curnumlen; s <= maxcolumnnumlen; s++) {
                    format += " ";
                }
                System.out.print(format + x * y);
            }
            System.out.println();
            Random r = new Random();
        }
    }

}
