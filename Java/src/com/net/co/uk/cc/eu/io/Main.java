package com.net.co.uk.cc.eu.io;

import java.util.Arrays;
import java.util.function.Function;

public class Main {

    public static void main(String[] args) {
        Boolean[] arr = {false, false, false};
        System.out.println(Arrays.stream(arr).anyMatch(v -> v));
    }
}
