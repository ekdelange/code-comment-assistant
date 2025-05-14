public class TestApp {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        int n = 10;
        int fib = fibonacci(n);
        System.out.println("Fibonacci number at position " + n + " is " + fib);
    }

    public static int fibonacci(int n) {
        if (n <= 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        } else {
            int a = 0;
            int b = 1;
            for (int i = 2; i <= n; i++) {
                int temp = a + b;
                a = b;
                b = temp;
            }
            return b;
        }
    }
}
