/**
 * This is the TestApp class, which serves as an entry point for the application.
 * The main functionality of this class is to print a greeting message and calculate
 * the Fibonacci number at a specified position.
 * 
 * The main method prints "Hello, World!" to the console and calculates the Fibonacci
 * number at position 10, displaying the result.
 * 
 * The fibonacci method is a static method that computes the Fibonacci number at a
 * given position using an iterative approach. It handles edge cases for positions
 * less than or equal to 0 and position 1.
 */
public class TestApp {

    /**
     * This is the main method which makes use of the TestApp class.
     * 
     * @param args Unused.
     */
    public static void main(String[] args) {
        // Your code here
    }
}/**
     * The main method serves as the entry point for the Java application.
     * It is called by the Java Virtual Machine (JVM) to start the execution of the program.
     *
     * @param args Command-line arguments passed to the program.
     */
    public static void main(String[] args) {        System.out.println("Hello, World!");
        int n = 10;
        int fib = fibonacci(n);
        System.out.println("Fibonacci number at position " + n + " is " + fib);
    }

/**
     * Calculates the nth Fibonacci number.
     *
     * The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones,
     * usually starting with 0 and 1. This method uses a recursive approach to calculate the nth number in the sequence.
     *
     * @param n the position in the Fibonacci sequence to calculate
     * @return the nth Fibonacci number
     */
    public static int fibonacci(int n) {        if (n <= 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        } else {
            int a = 0;
            int b = 1;
/**
 * Iterates from 2 to the given number n, inclusive.
 *
 * @param n the upper limit of the iteration (inclusive)
 */
for (int i = 2; i <= n; i++) {                int temp = a + b;
                a = b;
                b = temp;
            }
            return b;
        }
    }
}
